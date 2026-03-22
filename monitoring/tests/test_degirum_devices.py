from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from monitoring.degirum_devices import (
    benchmark_device_candidates,
    build_degirum_load_model_kwargs,
    coerce_optional_str,
    coerce_pathlike_to_str,
    detect_degirum_devices,
    parse_supported_device_types_from_error,
    resolve_degirum_runtime_target,
    sanitize_degirum_load_model_kwargs,
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
            if kwargs.get("device_type") != "TFLITE/CPU":
                raise RuntimeError("Supported device types are: ['TFLITE/CPU']")
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
    assert all(isinstance(call.get("zoo_url"), str) for call in captured)


def test_sanitize_degirum_load_model_kwargs_coerces_pathlike_fields() -> None:
    payload = sanitize_degirum_load_model_kwargs(
        {
            "model_name": "demo",
            "zoo_url": Path("/tmp/demo-zoo"),
            "inference_host_address": Path("@local"),
            "device_type": Path("TFLITE/CPU"),
        }
    )

    assert payload["zoo_url"] == "/tmp/demo-zoo"
    assert payload["inference_host_address"] == "@local"
    assert payload["device_type"] == "TFLITE/CPU"
    assert all(not isinstance(value, Path) for value in payload.values())


def test_build_kwargs_defaults_and_coerce_helpers() -> None:
    kwargs = build_degirum_load_model_kwargs(model_name="model-x", zoo_url=Path("/tmp/model-x"))

    assert kwargs["zoo_url"] == "/tmp/model-x"
    assert kwargs["inference_host_address"] == "@local"
    assert coerce_pathlike_to_str(Path("/tmp/a")) == "/tmp/a"
    assert coerce_optional_str(Path("/tmp/b")) == "/tmp/b"


def test_build_kwargs_omits_empty_device_type() -> None:
    kwargs = build_degirum_load_model_kwargs(
        model_name="model-x",
        zoo_url=Path("/tmp/model-x"),
        device_type="",
    )

    assert "device_type" not in kwargs
