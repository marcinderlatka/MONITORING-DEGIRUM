from __future__ import annotations

import itertools

from monitoring.degirum_devices import (
    benchmark_device_candidates,
    choose_best_degirum_device,
    detect_degirum_devices,
)


class _FakeModel:
    def close(self) -> None:
        return None

    def predict(self, sample) -> dict[str, object]:
        return {"ok": True, "sample": sample}


def test_detect_degirum_devices_always_returns_cpu_and_auto() -> None:
    class _NoApi:
        pass

    result = detect_degirum_devices(_NoApi())
    assert [item["id"] for item in result] == ["auto", "cpu"]
    assert result[1]["label"] == "CPU (procesor)"
    assert result[0]["label"] == "Auto"


def test_detect_degirum_devices_adds_gpu_only_after_positive_probe() -> None:
    class _Api:
        @staticmethod
        def enumerate_devices():
            raise RuntimeError("enumeration failed")

        @staticmethod
        def load_model(**kwargs):
            if kwargs.get("device") == "gpu":
                return _FakeModel()
            raise RuntimeError("unsupported")

    result = detect_degirum_devices(_Api())
    assert [item["id"] for item in result] == ["auto", "cpu", "gpu"]
    assert result[2]["label"] == "GPU (karta graficzna)"


def test_detect_degirum_devices_swallows_probe_errors() -> None:
    class _Api:
        @staticmethod
        def enumerate_devices():
            return []

        @staticmethod
        def load_model(**kwargs):
            raise RuntimeError("boom")

    result = detect_degirum_devices(_Api())
    assert [item["id"] for item in result] == ["auto", "cpu"]


def test_benchmark_device_candidates_updates_config() -> None:
    class _Api:
        @staticmethod
        def load_model(**kwargs):
            if kwargs.get("device_type") in {"gpu", "cpu"} or kwargs.get("device") in {"gpu", "cpu"}:
                return _FakeModel()
            raise RuntimeError("unsupported")

    config: dict[str, object] = {}
    result = benchmark_device_candidates(
        _Api(),
        model_name="dummy",
        candidates=["gpu", "cpu"],
        sample_input={"frame": b"x"},
        config=config,
    )

    assert [row["device"] for row in result["results"]] == ["gpu", "cpu"]
    assert config["degirum_available_devices"] == ["gpu", "cpu"]
    assert "degirum_last_benchmark" in config


def test_benchmark_device_candidates_is_deterministic_with_mocked_perf_counter(monkeypatch) -> None:
    class _Api:
        @staticmethod
        def load_model(**kwargs):
            if kwargs.get("device_type") in {"gpu", "cpu"} or kwargs.get("device") in {"gpu", "cpu"}:
                return _FakeModel()
            raise RuntimeError("unsupported")

    ticks = itertools.count(start=0, step=1)
    monkeypatch.setattr("monitoring.degirum_devices.time.perf_counter", lambda: next(ticks) / 1000.0)

    result = benchmark_device_candidates(
        _Api(),
        model_name="dummy",
        candidates=["gpu", "cpu"],
        sample_input={"frame": b"x"},
        inference_runs=2,
    )

    assert [row["device"] for row in result["results"]] == ["gpu", "cpu"]
    assert result["results"][0]["load_time_ms"] == 1.0
    assert result["results"][1]["load_time_ms"] == 1.0
    assert result["results"][0]["inference_time_ms"] == [1.0, 1.0]
    assert result["results"][1]["inference_time_ms"] == [1.0, 1.0]


def test_choose_best_degirum_device_falls_back_to_cpu_when_gpu_unstable() -> None:
    class _UnstableGpuModel(_FakeModel):
        def predict(self, sample) -> dict[str, object]:
            raise RuntimeError("gpu unstable")

    class _Api:
        @staticmethod
        def load_model(**kwargs):
            device = kwargs.get("device_type") or kwargs.get("device")
            if device == "gpu":
                return _UnstableGpuModel()
            if device == "cpu":
                return _FakeModel()
            raise RuntimeError("unsupported")

    config: dict[str, object] = {}
    selected = choose_best_degirum_device(
        _Api(),
        model_name="dummy",
        candidates=["gpu", "cpu"],
        sample_input={"frame": b"x"},
        config=config,
        auto_select=True,
    )

    assert selected == "cpu"
    assert config["degirum_preferred_device"] == "cpu"


def test_benchmark_never_passes_logical_cpu_gpu_as_device_type() -> None:
    captured: list[dict[str, object]] = []

    class _Api:
        @staticmethod
        def load_model(**kwargs):
            captured.append(dict(kwargs))
            device_type = kwargs.get("device_type")
            if device_type in {"cpu", "gpu"}:
                raise AssertionError("invalid logical device_type passed to load_model")
            return _FakeModel()

    benchmark_device_candidates(
        _Api(),
        model_name="dummy",
        candidates=["cpu", "gpu"],
        sample_input={"frame": b"x"},
    )

    assert captured
    assert all(call.get("device_type") not in {"cpu", "gpu"} for call in captured)
