"""Helpers for resilient DeGirum device detection.

This module isolates all probing logic so UI code can call one function and
always receive a safe, normalized list of device options.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .config import DEFAULT_MODEL, MODELS_PATH, is_valid_degirum_device_type, normalize_degirum_device_selection


GUI_LABELS = {
    "auto": "Auto",
    "cpu": "CPU (procesor)",
    "gpu": "GPU (karta graficzna)",
}

logger = logging.getLogger(__name__)


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
            attempts = [
                {
                    "model_name": model_name,
                    "inference_host_address": host,
                    "zoo_url": zoo_url,
                }
            ]
            if is_valid_degirum_device_type(device):
                attempts.append(
                    {
                        "model_name": model_name,
                        "inference_host_address": host,
                        "zoo_url": zoo_url,
                        "device_type": device,
                    }
                )
            else:
                attempts.append(
                    {
                        "model_name": model_name,
                        "inference_host_address": host,
                        "zoo_url": zoo_url,
                        "device": device,
                    }
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


def _load_model_for_device(
    dg_module: Any,
    *,
    model_name: str,
    zoo_url: str,
    candidate_hosts: Sequence[str],
    device: str,
) -> tuple[Any | None, str | None, str | None]:
    load_model = getattr(dg_module, "load_model", None)
    if not callable(load_model):
        return None, None, "Brak API load_model."

    last_error: str | None = None
    for host in candidate_hosts:
        attempts = [
            {
                "model_name": model_name,
                "inference_host_address": host,
                "zoo_url": zoo_url,
            },
            {
                "model_name": model_name,
                "zoo_url": zoo_url,
            },
        ]
        if is_valid_degirum_device_type(device):
            attempts.extend(
                [
                    {
                        "model_name": model_name,
                        "inference_host_address": host,
                        "zoo_url": zoo_url,
                        "device_type": device,
                    },
                    {
                        "model_name": model_name,
                        "zoo_url": zoo_url,
                        "device_type": device,
                    },
                ]
            )
        else:
            attempts.extend(
                [
                    {
                        "model_name": model_name,
                        "inference_host_address": host,
                        "zoo_url": zoo_url,
                        "device": device,
                    },
                    {
                        "model_name": model_name,
                        "zoo_url": zoo_url,
                        "device": device,
                    },
                ]
            )
        for kwargs in attempts:
            try:
                model = load_model(**kwargs)
                return model, host, None
            except Exception as exc:
                last_error = str(exc) or exc.__class__.__name__
    return None, None, last_error or "Nieudane ładowanie modelu."


def _run_short_inference(
    model: Any,
    sample_input: Any,
    *,
    runs: int,
) -> tuple[list[float], str | None]:
    if runs <= 0:
        return [], None

    infer_method = None
    for name in ("predict", "infer", "run"):
        maybe = getattr(model, name, None)
        if callable(maybe):
            infer_method = maybe
            break

    if infer_method is None:
        return [], "Model nie udostępnia predict/infer/run."

    times: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        infer_method(sample_input)
        times.append(time.perf_counter() - start)
    return times, None


def benchmark_device_candidates(
    dg_module: Any,
    *,
    model_name: str,
    candidates: Sequence[str],
    zoo_url: str | Path | None = None,
    candidate_hosts: Sequence[str] = ("@local", "localhost", "127.0.0.1"),
    sample_input: Any = None,
    inference_runs: int = 2,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Benchmark candidate devices by model load time and short inference runs."""

    normalized_zoo = str(Path(zoo_url) if zoo_url is not None else MODELS_PATH / model_name)
    normalized_candidates = [
        normalize_degirum_device_selection(item)
        for item in candidates
        if normalize_degirum_device_selection(item) not in {"inherit", "auto"}
    ]
    safe_runs = max(1, min(3, int(inference_runs)))

    logger.info(
        "Start benchmarku DeGirum: model=%s, kandydaci=%s, runs=%s",
        model_name,
        normalized_candidates,
        safe_runs,
    )

    results: list[dict[str, Any]] = []
    for candidate in normalized_candidates:
        entry: dict[str, Any] = {
            "device": candidate,
            "kind": _normalize_kind(candidate) or candidate,
            "available": False,
            "stable": False,
            "load_time_ms": None,
            "inference_time_ms": [],
            "mean_inference_time_ms": None,
            "score": 0.0,
            "error": None,
        }
        load_start = time.perf_counter()
        model = None
        try:
            model, host, load_error = _load_model_for_device(
                dg_module,
                model_name=model_name,
                zoo_url=normalized_zoo,
                candidate_hosts=tuple(candidate_hosts),
                device=candidate,
            )
            entry["host"] = host
            if model is None:
                entry["error"] = load_error
                results.append(entry)
                continue
            entry["load_time_ms"] = round((time.perf_counter() - load_start) * 1000.0, 3)
            entry["available"] = True

            infer_times, infer_error = _run_short_inference(model, sample_input, runs=safe_runs)
            entry["inference_time_ms"] = [round(t * 1000.0, 3) for t in infer_times]
            if infer_times:
                entry["mean_inference_time_ms"] = round((sum(infer_times) / len(infer_times)) * 1000.0, 3)
            entry["stable"] = infer_error is None
            if infer_error:
                entry["error"] = infer_error

            load_ms = float(entry["load_time_ms"] or 0.0)
            infer_ms = float(entry["mean_inference_time_ms"] or 0.0)
            base = 1_000_000.0 / (1.0 + load_ms + infer_ms)
            if entry["kind"] == "gpu":
                base += 1_000.0
            if entry["stable"]:
                base += 100.0
            entry["score"] = round(base, 3)
        except Exception as exc:
            entry["error"] = str(exc) or exc.__class__.__name__
        finally:
            close_method = getattr(model, "close", None)
            if callable(close_method):
                try:
                    close_method()
                except Exception:
                    pass
        results.append(entry)

    results.sort(
        key=lambda item: (
            item["available"],
            item["stable"],
            item.get("kind") == "gpu",
            item["score"],
        ),
        reverse=True,
    )
    available_devices = [item["device"] for item in results if item["available"]]
    payload = {
        "model_name": model_name,
        "results": results,
        "available_devices": available_devices,
    }

    if config is not None:
        config["degirum_last_benchmark"] = payload
        config["degirum_available_devices"] = available_devices

    logger.info("Wyniki benchmarku DeGirum: %s", results)
    return payload


def choose_best_degirum_device(
    dg_module: Any,
    *,
    model_name: str,
    candidates: Sequence[str],
    zoo_url: str | Path | None = None,
    candidate_hosts: Sequence[str] = ("@local", "localhost", "127.0.0.1"),
    sample_input: Any = None,
    inference_runs: int = 2,
    config: dict[str, Any] | None = None,
    auto_select: bool = False,
) -> str:
    """Choose best DeGirum device, prefer stable GPU and always keep CPU fallback."""

    benchmark = benchmark_device_candidates(
        dg_module,
        model_name=model_name,
        candidates=candidates,
        zoo_url=zoo_url,
        candidate_hosts=candidate_hosts,
        sample_input=sample_input,
        inference_runs=inference_runs,
        config=config,
    )

    choice = "cpu"
    rows = benchmark.get("results", [])
    stable_gpu = next(
        (
            row
            for row in rows
            if row.get("kind") == "gpu" and row.get("available") and row.get("stable")
        ),
        None,
    )
    stable_cpu = next(
        (
            row
            for row in rows
            if row.get("kind") == "cpu" and row.get("available") and row.get("stable")
        ),
        None,
    )

    if stable_gpu is not None:
        choice = str(stable_gpu.get("device") or "gpu")
    elif stable_cpu is not None:
        choice = str(stable_cpu.get("device") or "cpu")

    if config is not None and auto_select:
        config["degirum_preferred_device"] = choice

    logger.info("Rekomendowane urządzenie DeGirum: %s", choice)
    return choice


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
