"""Helpers for resilient DeGirum device detection.

This module isolates all probing logic so UI code can call one function and
always receive a safe, normalized list of device options.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .config import DEFAULT_MODEL, MODELS_PATH


GUI_LABELS = {
    "auto": "Auto",
    "cpu": "CPU (procesor)",
    "gpu": "GPU (karta graficzna)",
}


def _build_entry(
    *,
    device_id: str,
    kind: str,
    available: bool,
    details: str,
    score: float,
    recommended: bool,
) -> dict[str, Any]:
    return {
        "id": device_id,
        "label": GUI_LABELS.get(device_id, device_id.upper()),
        "kind": kind,
        "available": bool(available),
        "details": details,
        "score": float(score),
        "recommended": bool(recommended),
    }


def _safe_to_iter(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, dict)):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _normalize_kind(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if "gpu" in text or "cuda" in text or "opencl" in text:
        return "gpu"
    if "cpu" in text:
        return "cpu"
    if "auto" in text:
        return "auto"
    return text


def _enumerate_kinds(dg_module: Any) -> list[str]:
    kinds: list[str] = []
    for method_name in (
        "list_devices",
        "enumerate_devices",
        "get_available_devices",
        "get_supported_devices",
    ):
        method = getattr(dg_module, method_name, None)
        if not callable(method):
            continue
        try:
            rows = _safe_to_iter(method())
        except Exception:
            continue
        for row in rows:
            if isinstance(row, dict):
                for key in ("kind", "type", "device", "name"):
                    value = row.get(key)
                    normalized = _normalize_kind(value)
                    if normalized:
                        kinds.append(normalized)
            else:
                normalized = _normalize_kind(row)
                if normalized:
                    kinds.append(normalized)
        if kinds:
            break
    return kinds


def _probe_gpu_with_load_model(
    dg_module: Any,
    *,
    model_name: str,
    zoo_url: str,
    candidate_hosts: Sequence[str],
    candidate_devices: Sequence[str],
) -> tuple[bool, str]:
    load_model = getattr(dg_module, "load_model", None)
    if not callable(load_model):
        return False, "Brak API load_model do aktywnego probingu GPU."

    for host in candidate_hosts:
        for device in candidate_devices:
            attempts = (
                {
                    "model_name": model_name,
                    "inference_host_address": host,
                    "zoo_url": zoo_url,
                    "device_type": device,
                },
                {
                    "model_name": model_name,
                    "inference_host_address": host,
                    "zoo_url": zoo_url,
                    "device": device,
                },
                {
                    "model_name": model_name,
                    "zoo_url": zoo_url,
                    "device_type": device,
                },
            )
            for kwargs in attempts:
                try:
                    model = load_model(**kwargs)
                    close_method = getattr(model, "close", None)
                    if callable(close_method):
                        try:
                            close_method()
                        except Exception:
                            pass
                    return True, f"GPU wykryto przez load_model(host={host}, device={device})."
                except Exception as exc:
                    last_error = str(exc) or exc.__class__.__name__
                    continue
    return False, f"Probing GPU nie powiódł się ({last_error if 'last_error' in locals() else 'brak odpowiedzi'})."


def detect_degirum_devices(
    dg_module: Any,
    *,
    model_name: str = DEFAULT_MODEL,
    zoo_url: str | Path | None = None,
    candidate_hosts: Sequence[str] = ("@local", "localhost", "127.0.0.1"),
    candidate_devices: Sequence[str] = ("gpu", "cuda", "opencl"),
) -> list[dict[str, Any]]:
    """Detect available DeGirum devices and return normalized GUI records.

    Always returns at least ``auto`` and ``cpu`` entries. Any probing exception
    is captured locally and reflected only in the ``details`` field.
    """

    records: list[dict[str, Any]] = [
        _build_entry(
            device_id="auto",
            kind="auto",
            available=True,
            details="Automatyczny wybór urządzenia.",
            score=0.8,
            recommended=False,
        ),
        _build_entry(
            device_id="cpu",
            kind="cpu",
            available=True,
            details="Bezpieczny fallback procesora.",
            score=0.5,
            recommended=True,
        ),
    ]

    normalized_zoo = str(Path(zoo_url) if zoo_url is not None else MODELS_PATH / model_name)

    try:
        kinds = _enumerate_kinds(dg_module)
    except Exception as exc:
        kinds = []
        records[1]["details"] = f"Fallback CPU; błąd enumeracji: {exc}"

    gpu_available = any(kind == "gpu" for kind in kinds)
    gpu_details = "GPU znalezione przez API enumeracji urządzeń."

    if not gpu_available:
        try:
            gpu_available, gpu_details = _probe_gpu_with_load_model(
                dg_module,
                model_name=model_name,
                zoo_url=normalized_zoo,
                candidate_hosts=tuple(candidate_hosts),
                candidate_devices=tuple(candidate_devices),
            )
        except Exception as exc:
            gpu_available = False
            gpu_details = f"Probing GPU przerwany bezpiecznie: {exc}"

    if gpu_available:
        records.append(
            _build_entry(
                device_id="gpu",
                kind="gpu",
                available=True,
                details=gpu_details,
                score=0.95,
                recommended=True,
            )
        )
        records[1]["recommended"] = False

    return records
