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
