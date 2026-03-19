from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from monitoring.runtime_helpers import build_root_cause_summary, scale_bbox, stabilized_stream_fps


def test_stabilized_stream_fps_limits_aggressive_jump_against_fallback():
    samples = [29.8, 30.0, 30.2, 58.0, 60.0, 59.5, 30.1, 29.9]
    stabilized = stabilized_stream_fps(samples, fallback=30.0, min_samples=5, min_window_seconds=0.0)
    assert 20.0 <= stabilized <= 40.5


def test_stabilized_stream_fps_uses_min_window_gate():
    # ~0.4s of total samples => should keep fallback for min_window_seconds=1.0
    samples = [25.0, 25.0, 25.0, 25.0, 25.0]
    stabilized = stabilized_stream_fps(samples, fallback=20.0, min_samples=5, min_window_seconds=1.0)
    assert stabilized == 20.0


def test_build_root_cause_summary_prefers_new_bottleneck_labels():
    summary = build_root_cause_summary(
        ui_render_ms=42.0,
        ui_render_limit_ms=14.0,
        queue_size=31,
        queue_limit=10,
        infer_fps=2.0,
        detect_fps_target=8.0,
        stream_fps=3.0,
        writer_fps=10.0,
    )
    assert "gui_bottleneck" in summary
    assert "recording_bottleneck" in summary
    assert "inference_bottleneck" in summary
    assert "stream_bottleneck" in summary


def test_scale_bbox_handles_source_size_scaling():
    bbox = [64, 128, 320, 512]
    scaled = scale_bbox(bbox, frame_shape=(1080, 1920, 3), source_size=(640, 640))
    assert scaled == (192, 216, 960, 864)
