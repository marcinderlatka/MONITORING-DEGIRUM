from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from monitoring.degirum_devices import (
    benchmark_device_candidates,
    detect_degirum_devices,
    parse_supported_device_types_from_error,
    resolve_degirum_runtime_target,
)


class _FakeModel:
    def close(self) -> None:
        return None

    def predict(self, sample) -> dict[str, object]:
        return {"ok": True, "sample": sample}


def test_parse_supported_device_types_from_error() -> None:
    exc = RuntimeError("Unsupported. Supported device types are: ['TFLITE/CPU', 'TFLITE/GPU']")
    assert parse_supported_device_types_from_error(exc) == ["TFLITE/CPU", "TFLITE/GPU"]


def test_detect_degirum_devices_uses_supported_types_from_model_error() -> None:
    class _Api:
        @staticmethod
        def load_model(**kwargs):
            raise RuntimeError("Supported device types are: ['TFLITE/CPU']")

    result = detect_degirum_devices(_Api(), model_name="dummy")
    ids = [item["id"] for item in result]

    assert "cpu" in ids
    assert "gpu" in ids
    assert "TFLITE/CPU" in ids


def test_runtime_resolution_gpu_falls_back_to_cpu_when_model_supports_only_cpu() -> None:
    resolved = resolve_degirum_runtime_target(
        logical_selection="gpu",
        supported_device_types=["TFLITE/CPU"],
    )

    assert resolved["logical_selection"] == "gpu"
    assert resolved["final_device_type"] == "TFLITE/CPU"
    assert resolved["fallback_used"] is True


def test_benchmark_device_candidates_never_uses_logical_cpu_gpu_as_device_type() -> None:
    captured: list[dict[str, object]] = []

    class _Api:
        @staticmethod
        def load_model(**kwargs):
            captured.append(dict(kwargs))
            if kwargs.get("device_type") == "__INVALID__/__INVALID__":
                raise RuntimeError("Supported device types are: ['TFLITE/CPU']")
            if kwargs.get("device_type") != "TFLITE/CPU":
                raise AssertionError("invalid runtime device type")
            return _FakeModel()

    benchmark = benchmark_device_candidates(
        _Api(),
        model_name="dummy",
        candidates=["cpu", "gpu", "auto"],
        sample_input={"frame": b"x"},
    )

    assert captured
    assert all(call.get("device_type") not in {"cpu", "gpu"} for call in captured)
    assert benchmark["supported_device_types"] == ["TFLITE/CPU"]
