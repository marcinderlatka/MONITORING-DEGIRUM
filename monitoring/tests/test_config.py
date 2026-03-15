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
