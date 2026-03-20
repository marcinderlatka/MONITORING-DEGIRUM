from __future__ import annotations

from monitoring.degirum_devices import detect_degirum_devices


class _FakeModel:
    def close(self) -> None:
        return None


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
            if kwargs.get("device_type") == "gpu":
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
