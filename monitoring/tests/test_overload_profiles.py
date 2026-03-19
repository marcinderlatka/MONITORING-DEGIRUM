from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from monitoring.runtime_helpers import evaluate_overload_transition, overload_level_profile


def test_overload_level_profiles_match_multilevel_policy():
    level1 = overload_level_profile(1)
    assert level1.performance_log_interval_s > overload_level_profile(0).performance_log_interval_s
    assert level1.thumb_preview_fps_factor == 1.0
    assert level1.detect_fps_factor == 1.0

    level2 = overload_level_profile(2)
    assert level2.thumb_preview_fps_factor < 1.0
    assert level2.disable_nonessential_overlays is True
    assert level2.detect_fps_factor == 1.0

    level3 = overload_level_profile(3)
    assert level3.detect_fps_factor < 1.0
    assert level3.thumb_preview_fps_factor <= level2.thumb_preview_fps_factor


def test_overload_hysteresis_uses_different_enter_and_exit_thresholds():
    level, ts, reason = evaluate_overload_transition(
        now_ts=5.0,
        active_camera_count=5,
        gui_load_fps=65.0,
        recording_count=0,
        currently_level=0,
        last_change_ts=0.0,
        protection_enabled=True,
        min_camera_count=2,
        camera_threshold=4,
        load_per_camera_threshold=10.0,
        enter_debounce_seconds=1.0,
        exit_debounce_seconds=1.0,
        ui_render_ms=8.0,
        max_ui_render_ms=14.0,
        queue_size=2,
        max_queue_size=24,
        preview_bandwidth_mbps=2.0,
        max_preview_bandwidth_mbps=12.0,
    )
    assert level == 2
    assert reason == "condition-stable-enter-L2"

    # Slightly reduced load should keep L2 (exit threshold is lower than enter).
    level2, ts2, reason2 = evaluate_overload_transition(
        now_ts=5.6,
        active_camera_count=5,
        gui_load_fps=56.0,
        recording_count=0,
        currently_level=level,
        last_change_ts=ts,
        protection_enabled=True,
        min_camera_count=2,
        camera_threshold=4,
        load_per_camera_threshold=10.0,
        enter_debounce_seconds=1.0,
        exit_debounce_seconds=1.0,
        ui_render_ms=8.0,
        max_ui_render_ms=14.0,
        queue_size=2,
        max_queue_size=24,
        preview_bandwidth_mbps=2.0,
        max_preview_bandwidth_mbps=12.0,
    )
    assert level2 == 2
    assert reason2 == "stable-L2"

    # Enough reduction should allow exiting to L1.
    level3, _ts3, reason3 = evaluate_overload_transition(
        now_ts=6.8,
        active_camera_count=5,
        gui_load_fps=40.0,
        recording_count=0,
        currently_level=level2,
        last_change_ts=ts2,
        protection_enabled=True,
        min_camera_count=2,
        camera_threshold=4,
        load_per_camera_threshold=10.0,
        enter_debounce_seconds=1.0,
        exit_debounce_seconds=1.0,
        ui_render_ms=6.0,
        max_ui_render_ms=14.0,
        queue_size=1,
        max_queue_size=24,
        preview_bandwidth_mbps=1.0,
        max_preview_bandwidth_mbps=12.0,
    )
    assert level3 == 1
    assert reason3 == "condition-stable-exit-L1"
