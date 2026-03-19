import json
from pathlib import Path


def test_fill_camera_defaults_normalises_record_path(tmp_path, monkeypatch):
    from monitoring import config

    monkeypatch.setattr(config, "BASE_DIR", tmp_path)
    monkeypatch.setattr(config, "DEFAULT_RECORD_PATH", tmp_path / "nagrania")

    camera = {"name": "Cam", "record_path": "./relative"}

    updated = config.fill_camera_defaults(camera)

    assert Path(updated["record_path"]) == (tmp_path / "relative").resolve()


def test_normalise_catalog_entry_uses_project_root(tmp_path, monkeypatch):
    from monitoring import storage

    catalog_path = tmp_path / "recordings_catalog.json"
    catalog_path.write_text(
        json.dumps(
            [
                {
                    "camera": "Cam",
                    "filepath": "./relative/video.mp4",
                }
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(storage, "BASE_DIR", tmp_path)

    entries = storage.load_recordings_catalog(catalog_path)

    assert entries[0]["filepath"] == str((tmp_path / "relative" / "video.mp4").resolve())


def test_config_backward_compat_threshold_mapping():
    from monitoring import config

    camera = {"name": "Cam", "confidence_threshold": 0.67}
    updated = config.fill_camera_defaults(camera)

    assert updated["confidence_threshold_draw"] == 0.67
    assert updated["confidence_threshold_record"] == 0.67
    assert updated["thumbnail_mode"] == "first_detection"


def test_config_backward_compat_legacy_confidence_fallback_from_record_threshold():
    from monitoring import config

    camera = {"name": "Cam", "confidence_threshold_record": 0.61}
    updated = config.fill_camera_defaults(camera)

    assert updated["confidence_threshold"] == 0.61
    assert updated["confidence_threshold_draw"] == config.DEFAULT_CONFIDENCE_THRESHOLD
    assert updated["confidence_threshold_record"] == 0.61


def test_fill_camera_defaults_adds_reliability_defaults():
    from monitoring import config

    camera = {"name": "Cam"}
    updated = config.fill_camera_defaults(camera)

    assert updated["required_misses_to_end_detection"] == 3
    assert updated["min_record_seconds"] == 3
    assert updated["sensitivity_profile"] == "balanced"
    assert updated["recording_backend"] == "current"


def test_fill_camera_defaults_normalizes_recording_backend():
    from monitoring import config

    camera = {"name": "Cam", "recording_backend": "FFMPEG"}
    updated = config.fill_camera_defaults(camera)
    assert updated["recording_backend"] == "ffmpeg"

    fallback = config.fill_camera_defaults({"name": "Cam2", "recording_backend": "unknown"})
    assert fallback["recording_backend"] == "current"


def test_fill_camera_defaults_applies_selected_sensitivity_profile():
    from monitoring import config

    camera = {"name": "Cam", "sensitivity_profile": "high_precision"}
    updated = config.fill_camera_defaults(camera)

    assert updated["confidence_threshold_record"] == config.SENSITIVITY_PROFILES["high_precision"]["confidence_threshold_record"]
    assert updated["required_hits_to_start_recording"] == 3


def test_preview_role_defaults_and_config_fill():
    from monitoring import config

    camera = {"name": "Cam"}
    updated = config.fill_camera_defaults(camera)

    assert updated["preview_fps_main"] == 12
    assert updated["preview_fps_grid"] == 3
    assert updated["preview_fps_thumb"] == 2
    assert updated["preview_pause_when_hidden"] is True
    assert updated["preview_main_max_width"] == 1280
    assert updated["preview_main_max_height"] == 720
    assert updated["preview_grid_max_width"] == 640
    assert updated["preview_grid_max_height"] == 360
    assert updated["preview_thumb_max_width"] == 320
    assert updated["preview_thumb_max_height"] == 180
    assert updated["overlay_text_enabled"] is True
    assert updated["overlay_draw_every_n"] == 2
    assert updated["camera_priority"] == "normal"
    assert "preview_channel_policies" in updated
    assert updated["preview_channel_policies"]["grid"]["fps"] == 3.0


def test_overload_config_backward_compat(tmp_path):
    from monitoring.config import load_config

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text('{"cameras":[{"name":"c1","rtsp":"x"}]}', encoding="utf-8")
    cfg = load_config(cfg_path)

    assert cfg["overload_protection_enabled"] is True
    assert cfg["overload_camera_count_threshold"] == 6
    assert cfg["overload_reduce_thumb_preview_fps"] == 1
    assert cfg["quality_performance_preset"] == "economy_monitoring"
    assert cfg["grid_preview_quality"] == "normal"
    assert cfg["config_watchdog_enabled"] is True
    assert "log_filters" in cfg
    assert "groups" in cfg["log_filters"]


def test_quality_monitoring_preset_uses_higher_grid_limits():
    from monitoring.config import QUALITY_PERFORMANCE_PRESETS

    quality = QUALITY_PERFORMANCE_PRESETS["quality_monitoring"]
    assert quality["preview_grid_max_width"] >= 1280
    assert quality["preview_grid_max_height"] >= 720


def test_log_filters_are_normalized_and_applied():
    from monitoring import config

    filters = config.normalize_log_filters(
        {
            "groups": ["detection", "error", "detection"],
            "levels": ["warning", "ERROR"],
            "sources": ["worker", "app", "worker"],
        }
    )
    assert filters["groups"] == ["detection", "error"]
    assert filters["levels"] == ["WARNING", "ERROR"]
    assert filters["sources"] == ["worker", "app"]


def test_log_filter_source_category_mapping(monkeypatch):
    from monitoring import config

    monkeypatch.setattr(
        config,
        "LOG_FILTERS",
        {"groups": ["application"], "levels": ["INFO"], "sources": ["app"]},
    )

    assert config.is_log_entry_enabled("application", "INFO", "monitoring.runtime_helpers") is True
    assert config.is_log_entry_enabled("application", "INFO", "ui") is False


def test_restart_required_for_model_or_rtsp_url_change():
    from monitoring.runtime_helpers import classify_camera_setting_changes

    old = {"name": "Cam", "model": "a", "rtsp": "rtsp://1", "fps": 10}
    new = {"name": "Cam", "model": "b", "rtsp": "rtsp://2", "fps": 12}
    changed, restart = classify_camera_setting_changes(old, new, {"model", "rtsp", "type"})

    assert "model" in changed
    assert "rtsp" in changed
    assert "fps" in changed
    assert set(restart) == {"model", "rtsp"}


def test_classify_camera_changed_keys_helper_for_live_fields():
    from monitoring.runtime_helpers import classify_camera_setting_changes

    old = {"name": "Cam", "fps": 10, "draw_overlays": True}
    new = {"name": "Cam", "fps": 15, "draw_overlays": False}
    changed, restart = classify_camera_setting_changes(old, new, {"model", "rtsp"})

    assert set(changed) == {"draw_overlays", "fps"}
    assert restart == []


def test_fill_camera_defaults_adds_info_overlay_flag():
    from monitoring import config

    camera = {"name": "Cam"}
    updated = config.fill_camera_defaults(camera)

    assert updated["show_camera_info_overlay"] is True


def test_fill_camera_defaults_adds_thumbnail_overlay_style_defaults():
    from monitoring import config

    camera = {"name": "Cam"}
    updated = config.fill_camera_defaults(camera)

    assert updated["thumbnail_overlay_enabled"] is True
    assert updated["thumbnail_box_thickness"] == 1
    assert updated["thumbnail_font_scale"] == 0.5
    assert updated["thumbnail_font_thickness"] == 1
