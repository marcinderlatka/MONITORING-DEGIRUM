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


def test_fill_camera_defaults_adds_reliability_defaults():
    from monitoring import config

    camera = {"name": "Cam"}
    updated = config.fill_camera_defaults(camera)

    assert updated["required_misses_to_end_detection"] == 1
    assert updated["min_record_seconds"] == 3


def test_preview_role_defaults_and_config_fill():
    from monitoring import config

    camera = {"name": "Cam"}
    updated = config.fill_camera_defaults(camera)

    assert updated["preview_fps_main"] == 15
    assert updated["preview_fps_thumb"] == 3
    assert updated["preview_pause_when_hidden"] is True


def test_overload_config_backward_compat(tmp_path):
    from monitoring.config import load_config

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text('{"cameras":[{"name":"c1","rtsp":"x"}]}', encoding="utf-8")
    cfg = load_config(cfg_path)

    assert cfg["overload_protection_enabled"] is True
    assert cfg["overload_camera_count_threshold"] == 6
    assert cfg["overload_reduce_thumb_preview_fps"] == 1


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
