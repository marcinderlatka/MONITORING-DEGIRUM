from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from monitoring import storage


def test_recording_metadata_io_worker_retries_and_logs(monkeypatch):
    attempts = {"count": 0}
    logs: list[str] = []

    def flaky(_meta: dict) -> None:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise OSError("dysk chwilowo zajęty")

    monkeypatch.setattr(storage, "_persist_recording_metadata_sync", flaky)
    monkeypatch.setattr(storage, "app_log", lambda *_a, **kwargs: logs.append(str(kwargs.get("details", ""))))

    worker = storage._RecordingMetadataIoWorker(max_retries=3, base_retry_delay_s=0.01)
    worker.submit({"filepath": "/tmp/sample.mp4", "camera": "Cam1"})

    assert worker.flush(timeout_s=2.0) is True
    assert attempts["count"] == 3
    assert any("próba=1/3" in item for item in logs)
    assert any("próba=2/3" in item for item in logs)
    assert worker.shutdown(timeout_s=2.0) is True


def test_flush_storage_flushes_background_io(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(storage._RECORDING_METADATA_IO, "flush", lambda timeout_s=8.0: calls.append("io"))
    monkeypatch.setattr(storage._RECORDINGS_CATALOG, "flush", lambda: calls.append("catalog"))

    storage.flush_storage()

    assert calls == ["io", "catalog"]
