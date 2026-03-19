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
    build_recording_sidecar_metadata,
    camera_name_for_path,
    default_filter_bounds,
    iter_recording_entries_progressive,
    load_history_metadata,
    load_recording_entries,
    merge_recording_entries,
)
from monitoring.runtime_helpers import camera_overlay_anchor, compute_effective_writer_fps, compute_letterboxed_rect, thumbnail_load_outcome
from monitoring.recordings import thumbnail_candidates_for_entry
from monitoring.recordings import alert_thumbnail_candidates_for_event


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
    assert payload["scene_thumb"] == "/tmp/a.mp4.jpg"
    assert payload["sensitivity_profile"] == "custom"
    assert payload["recorder_drop_rate"] == 0.0
    assert payload["recorder_queue_latency_proxy_s"] == 0.0
    assert payload["recorder_enqueue_stride"] == 1
    assert payload["recorder_degradation_level"] == 0
    assert payload["writer_fps_base"] == 0.0


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
    assert compute_effective_writer_fps(5, 3.0, 25.0) == 25.0
    assert compute_effective_writer_fps(0, 3.0, 0.0) == 3.0


def test_old_metadata_without_new_fields_still_loads(tmp_path):
    root = tmp_path / "Cam1"
    root.mkdir()
    video = root / "alert_20240102_030405.mp4"
    video.write_bytes(b"")
    video.with_suffix(".mp4.json").write_text(json.dumps({"label": "person", "time": "2024-01-02 03:04:05"}), encoding="utf-8")

    metadata = build_recording_metadata(str(video), [("Cam1", str(root))])

    assert metadata.label == "person"
    assert metadata.display_time == "2024-01-02 03:04:05"


def test_metadata_preserves_phase4_diagnostics_fields():
    payload = build_recording_sidecar_metadata(
        camera="Cam1",
        label="person",
        confidence=0.8,
        event_time="2024-01-01 00:00:00",
        filepath="/tmp/z.mp4",
        thumb="/tmp/z.jpg",
        source_fps=20.0,
        writer_fps=5.0,
        detect_fps=3.0,
        event_start_ts=1.0,
        thumbnail_ts=1.2,
        frames_written=10,
        dropped_frames=1,
        thumbnail_mode="first_detection",
        inference_count=4,
        positive_detection_count=2,
        preview_role_at_start="thumb",
        overload_degraded_at_start=True,
        measured_capture_fps=19.5,
        effective_detect_fps=2.8,
        preview_frames_dropped=33,
        skipped_inference_cycles=4,
    )
    assert payload["preview_role_at_start"] == "thumb"
    assert payload["overload_degraded_at_start"] is True
    assert payload["effective_detect_fps"] == 2.8
    assert payload["preview_frames_dropped"] == 33


def test_metadata_preserves_recorder_efficiency_metrics():
    payload = build_recording_sidecar_metadata(
        camera="Cam2",
        label="car",
        confidence=0.75,
        event_time="2024-01-01 00:00:00",
        filepath="/tmp/y.mp4",
        thumb="/tmp/y.jpg",
        source_fps=25.0,
        writer_fps=4.0,
        detect_fps=3.0,
        event_start_ts=2.0,
        thumbnail_ts=2.1,
        frames_written=20,
        dropped_frames=3,
        thumbnail_mode="best_detection",
        inference_count=12,
        positive_detection_count=6,
        recorder_drop_rate=0.13,
        recorder_queue_latency_proxy_s=2.5,
        recorder_enqueue_stride=3,
        recorder_degradation_level=2,
        writer_fps_base=6.0,
        stream_fps_measured=19.2,
        writer_fps_selected=6.0,
        writer_fps_reason="measured_stream",
    )
    assert payload["recorder_drop_rate"] == 0.13
    assert payload["recorder_queue_latency_proxy_s"] == 2.5
    assert payload["recorder_enqueue_stride"] == 3
    assert payload["recorder_degradation_level"] == 2
    assert payload["writer_fps_base"] == 6.0
    assert payload["stream_fps_measured"] == 19.2
    assert payload["writer_fps_selected"] == 6.0
    assert payload["writer_fps_reason"] == "measured_stream"


def test_load_recording_entries_uses_catalog_when_available(tmp_path, monkeypatch):
    cam = tmp_path / "Cam1"
    cam.mkdir()
    mp4 = cam / "a_20240101_010101.mp4"
    mp4.write_bytes(b"x")

    monkeypatch.setattr("monitoring.recordings.iter_catalog_entries", lambda *_a, **_k: [build_recording_metadata(str(mp4), [("Cam1", str(cam))])])
    monkeypatch.setattr("monitoring.recordings.discover_recordings", lambda *_a, **_k: [])

    entries, diag = load_recording_entries([("Cam1", str(cam))], [])
    assert len(entries) == 1
    assert entries[0].filepath == str(mp4.resolve())
    assert diag["used_disk_fallback"] is False


def test_load_recording_entries_falls_back_to_disk_scan(tmp_path, monkeypatch):
    cam = tmp_path / "Cam1"
    cam.mkdir()
    mp4 = cam / "b_20240101_010101.mp4"
    mp4.write_bytes(b"x")

    monkeypatch.setattr("monitoring.recordings.iter_catalog_entries", lambda *_a, **_k: [])
    monkeypatch.setattr("monitoring.recordings.discover_recordings", lambda *_a, **_k: [build_recording_metadata(str(mp4), [("Cam1", str(cam))])])

    entries, diag = load_recording_entries([("Cam1", str(cam))], [])
    assert len(entries) == 1
    assert diag["used_disk_fallback"] is True


def test_load_recording_entries_deduplicates_catalog_and_disk_entries(tmp_path):
    cam = tmp_path / "Cam1"
    cam.mkdir()
    mp4 = cam / "c_20240101_010101.mp4"
    mp4.write_bytes(b"x")
    catalog_entry = build_recording_metadata(str(mp4), [("Cam1", str(cam))], overrides={"label": "person"})
    disk_entry = build_recording_metadata(str(mp4), [("Cam1", str(cam))], overrides={"label": "car"})

    merged, disk_only = merge_recording_entries([catalog_entry], [disk_entry])
    assert len(merged) == 1
    assert disk_only == []
    assert merged[0].label == "person"


def test_default_filter_range_does_not_hide_old_recordings(tmp_path):
    cam = tmp_path / "Cam1"
    cam.mkdir()
    old = cam / "x_20220101_010101.mp4"
    new = cam / "x_20250101_010101.mp4"
    old.write_bytes(b"x")
    new.write_bytes(b"x")
    entries = [
        build_recording_metadata(str(old), [("Cam1", str(cam))]),
        build_recording_metadata(str(new), [("Cam1", str(cam))]),
    ]
    dfrom, dto = default_filter_bounds(entries)
    assert dfrom.year <= 2022
    assert dto.year >= 2025


def test_missing_catalog_entries_can_be_recovered_from_disk(tmp_path, monkeypatch):
    cam = tmp_path / "Cam1"
    cam.mkdir()
    mp4 = cam / "d_20240101_010101.mp4"
    mp4.write_bytes(b"x")

    healed: list[dict] = []
    monkeypatch.setattr("monitoring.recordings.iter_catalog_entries", lambda *_a, **_k: [])
    monkeypatch.setattr("monitoring.recordings.discover_recordings", lambda *_a, **_k: [build_recording_metadata(str(mp4), [("Cam1", str(cam))])])
    monkeypatch.setattr("monitoring.recordings.update_recordings_catalog", lambda payload: healed.append(payload))

    entries, _ = load_recording_entries([("Cam1", str(cam))], [], heal_catalog=True)
    assert len(entries) == 1
    assert len(healed) == 1
    assert healed[0]["filepath"] == str(mp4.resolve())


def test_load_recording_entries_emits_progress_chunks(tmp_path, monkeypatch):
    cam = tmp_path / "Cam1"
    cam.mkdir()
    mp4 = cam / "d_20240101_010101.mp4"
    mp4.write_bytes(b"x")

    seen: list[tuple[list, dict]] = []

    monkeypatch.setattr("monitoring.recordings.iter_catalog_entries", lambda *_a, **_k: [])
    monkeypatch.setattr("monitoring.recordings.discover_recordings", lambda *_a, **_k: [build_recording_metadata(str(mp4), [("Cam1", str(cam))])])

    entries, diagnostics = load_recording_entries(
        [("Cam1", str(cam))],
        [],
        chunk_size=1,
        on_chunk=lambda chunk, progress: seen.append((chunk, progress)),
    )

    assert len(entries) == 1
    assert entries[0].filepath == str(mp4.resolve())
    assert diagnostics
    assert seen
    assert any(progress.get("phase") == "final" for _chunk, progress in seen)


def test_alert_thumbnail_prefers_scene_preview_not_object_crop(tmp_path):
    video = tmp_path / "event.mp4"
    video.write_bytes(b"")
    candidates = alert_thumbnail_candidates_for_event(
        {
            "filepath": str(video),
            "alert_thumb": str(video) + ".alert.jpg",
            "thumb": str(video) + ".jpg",
        }
    )
    assert candidates[0].endswith(".alert.jpg")
    assert candidates[1].endswith(".scene.jpg") or candidates[1].endswith(".preview.jpg") or candidates[1].endswith(".jpg")


def test_alert_thumbnail_candidates_prefer_scene_thumb_over_alert_thumb(tmp_path):
    video = tmp_path / "event2.mp4"
    video.write_bytes(b"")
    candidates = alert_thumbnail_candidates_for_event(
        {
            "filepath": str(video),
            "scene_thumb": str(video) + ".scene.jpg",
            "alert_thumb": str(video) + ".alert.jpg",
        }
    )
    assert candidates[0].endswith(".scene.jpg")
    assert candidates[1].endswith(".alert.jpg")


def test_recording_thumbnail_candidate_order_prefers_explicit_jpg(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    metadata = build_recording_metadata(
        str(video),
        [("cam", str(tmp_path))],
        overrides={"thumb": str(tmp_path / "poster.jpg")},
    )
    candidates = thumbnail_candidates_for_entry(metadata)
    assert candidates[0] == str((tmp_path / "poster.jpg").resolve())


def test_tile_thumbnail_loading_resolves_to_success_or_fallback():
    class _NullImage:
        def isNull(self):
            return True

    class _ValidImage:
        def isNull(self):
            return False

    assert thumbnail_load_outcome(_NullImage()) == "fallback"
    assert thumbnail_load_outcome(_ValidImage()) == "success"
    assert thumbnail_load_outcome(None) == "fallback"


def test_camera_overlay_anchor_uses_visible_image_rect_not_whole_widget():
    image_rect = compute_letterboxed_rect(1920, 1080, 1000, 1000)
    # visible rect should be centered vertically (letterbox bars on top/bottom)
    assert image_rect == (0, 219, 1000, 562)
    anchor = camera_overlay_anchor(image_rect, (260, 120), padding=10)
    assert anchor[0] == 10
    assert anchor[1] == 651
