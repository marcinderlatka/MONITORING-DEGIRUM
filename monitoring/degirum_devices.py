"""Helpers for resilient DeGirum device detection and runtime resolution."""

from __future__ import annotations

import inspect
import ast
import logging
import os
import re
import time
from contextlib import suppress
from queue import Queue
from threading import Thread
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import DEFAULT_MODEL, MODELS_PATH, is_valid_degirum_device_type, normalize_degirum_device_selection

GUI_LABELS = {
    "auto": "Auto",
    "cpu": "CPU (procesor)",
    "gpu": "GPU (karta graficzna)",
}

logger = logging.getLogger(__name__)
_SUPPORTED_RE = re.compile(r"Supported device types are:\s*(\[[^\]]*\])", re.IGNORECASE)
_PROBE_FALLBACK_DEVICE_TYPES = (
    "CPU",
    "OPENVINO/CPU",
    "CUDA/GPU",
    "TENSORRT/GPU",
    "HAILORT/HAILO8",
    "AXELERA/GPU",
    "DEEPX/GPU",
)


def coerce_pathlike_to_str(value: object) -> object:
    """Convert ``Path``/``os.PathLike`` values to ``str`` for DeGirum APIs."""
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, os.PathLike):
        try:
            coerced = os.fspath(value)
            return str(coerced)
        except Exception:
            logger.warning("degirum load_model coercion failed for path-like value type=%s", type(value).__name__)
            return str(value)
    return value


def coerce_optional_str(value: object) -> str | None:
    """Normalize optional text values into ``str`` while preserving ``None``."""
    if value is None:
        return None
    normalized = coerce_pathlike_to_str(value)
    if isinstance(normalized, bytes):
        try:
            return normalized.decode("utf-8", errors="ignore")
        except Exception:
            return str(normalized)
    return str(normalized)


def sanitize_degirum_load_model_kwargs(kwargs: dict[str, object]) -> dict[str, object]:
    """Return kwargs safe for ``dg.load_model`` (no ``Path`` / ``PathLike`` values)."""
    sanitized: dict[str, object] = {}
    for key, value in kwargs.items():
        normalized = coerce_pathlike_to_str(value)
        if key in {"model_name", "zoo_url", "model_path", "inference_host_address", "device_type"}:
            normalized_opt = coerce_optional_str(normalized)
            if normalized_opt is None:
                continue
            normalized = normalized_opt
        elif isinstance(normalized, bytes):
            normalized = normalized.decode("utf-8", errors="ignore")
        sanitized[key] = normalized
    return sanitized


def build_degirum_load_model_kwargs(
    *,
    model_name: object,
    inference_host_address: object = "@local",
    zoo_url: object | None = None,
    device_type: object | None = None,
) -> dict[str, object]:
    """Build and sanitize canonical kwargs passed to ``dg.load_model``."""
    normalized_model = coerce_optional_str(model_name) or ""
    zoo_value = zoo_url if zoo_url is not None else MODELS_PATH / normalized_model
    payload: dict[str, object] = {
        "model_name": normalized_model,
        "inference_host_address": coerce_optional_str(inference_host_address) or "@local",
        "zoo_url": zoo_value,
    }
    normalized_device = coerce_optional_str(device_type) if device_type is not None else None
    if normalized_device:
        payload["device_type"] = normalized_device
    return sanitize_degirum_load_model_kwargs(payload)


def _log_load_model_diagnostics(prefix: str, raw_kwargs: dict[str, object], sanitized_kwargs: dict[str, object]) -> None:
    logger.debug("%s raw kwargs=%s", prefix, raw_kwargs)
    logger.debug(
        "%s sanitized types=%s",
        prefix,
        {key: type(value).__name__ for key, value in sanitized_kwargs.items()},
    )


def _normalize_supported_device_type(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.upper()
    if "/" not in text:
        return ""
    runtime, device = text.split("/", 1)
    runtime = runtime.strip()
    device = device.strip()
    if not runtime or not device:
        return ""
    return f"{runtime}/{device}"


def device_kind_from_type(device_type: str) -> str:
    normalized = _normalize_supported_device_type(device_type)
    if not normalized:
        return ""
    _, device = normalized.split("/", 1)
    device_low = device.lower()
    if any(token in device_low for token in ("gpu", "cuda", "opencl", "metal", "vulkan")):
        return "gpu"
    if "cpu" in device_low:
        return "cpu"
    return "other"


def parse_supported_device_types_from_error(error: object) -> list[str]:
    text = str(error or "")
    if not text:
        return []
    match = _SUPPORTED_RE.search(text)
    if not match:
        return []
    raw_list = match.group(1)
    try:
        parsed = ast.literal_eval(raw_list)
    except Exception:
        return []
    if not isinstance(parsed, (list, tuple)):
        return []
    normalized: list[str] = []
    for item in parsed:
        dev_type = _normalize_supported_device_type(item)
        if dev_type and dev_type not in normalized:
            normalized.append(dev_type)
    return normalized


def load_model_with_timeout(
    dg_module: Any,
    *,
    timeout_s: float,
    **load_kwargs: object,
) -> Any:
    """Run ``dg.load_model`` in a daemon thread and enforce a hard timeout."""
    load_model = getattr(dg_module, "load_model", None)
    if not callable(load_model):
        raise RuntimeError("dg.load_model is unavailable")

    result_queue: Queue[tuple[bool, Any]] = Queue(maxsize=1)

    def _runner() -> None:
        try:
            model = load_model(**sanitize_degirum_load_model_kwargs(dict(load_kwargs)))
        except Exception as exc:
            result_queue.put((False, exc))
            return
        result_queue.put((True, model))

    thread = Thread(target=_runner, name="dg-load-model-timeout", daemon=True)
    thread.start()
    thread.join(timeout=max(0.1, float(timeout_s)))
    if thread.is_alive():
        raise TimeoutError(f"dg.load_model timeout after {timeout_s:.1f}s")
    ok, payload = result_queue.get_nowait()
    if ok:
        return payload
    raise payload


def get_model_supported_device_types(
    dg_module: Any,
    *,
    model_name: str,
    zoo_url: str | Path | None = None,
    inference_host_address: str = "@local",
    cache: dict[tuple[str, str], list[str]] | None = None,
    supported_cache: dict[tuple[str, str], list[str]] | None = None,
) -> list[str]:
    if cache is None and supported_cache is not None:
        cache = supported_cache

    normalized_model = coerce_optional_str(model_name) or ""
    normalized_zoo = coerce_optional_str(zoo_url if zoo_url is not None else MODELS_PATH / normalized_model) or ""
    cache_key = (normalized_model, normalized_zoo)
    if cache is not None and cache_key in cache:
        return list(cache[cache_key])

    load_model = getattr(dg_module, "load_model", None)
    if not callable(load_model):
        if cache is not None:
            cache[cache_key] = []
        return []

    supported: list[str] = []
    probe_candidates: list[str] = []

    get_supported_devices = getattr(dg_module, "get_supported_devices", None)
    if callable(get_supported_devices):
        try:
            sig = inspect.signature(get_supported_devices)
            kwargs: dict[str, object] = {}
            if "inference_host_address" in sig.parameters:
                kwargs["inference_host_address"] = inference_host_address
            if "zoo_url" in sig.parameters:
                kwargs["zoo_url"] = normalized_zoo
            detected = get_supported_devices(**kwargs)
        except Exception as exc:
            logger.debug("degirum get_supported_devices failed error=%s", exc)
        else:
            for candidate in detected:
                normalized = _normalize_supported_device_type(candidate)
                if normalized and normalized not in probe_candidates:
                    probe_candidates.append(normalized)

    # Keep only real runtime/device strings for probing.
    for candidate in _PROBE_FALLBACK_DEVICE_TYPES:
        normalized = _normalize_supported_device_type(candidate)
        if normalized and normalized not in probe_candidates:
            probe_candidates.append(normalized)

    for device_type in probe_candidates:
        raw_probe_kwargs = {
            "model_name": normalized_model,
            "inference_host_address": inference_host_address,
            "zoo_url": normalized_zoo,
            "device_type": device_type,
        }
        probe_kwargs = sanitize_degirum_load_model_kwargs(raw_probe_kwargs)
        _log_load_model_diagnostics("degirum supported-probe", raw_probe_kwargs, probe_kwargs)
        model = None
        try:
            logger.debug("degirum supported-probe load attempt device_type=%s", device_type)
            model = load_model_with_timeout(dg_module, timeout_s=6.0, **probe_kwargs)
            logger.debug("degirum supported-probe load success device_type=%s", device_type)
            if device_type not in supported:
                supported.append(device_type)
        except TimeoutError:
            logger.warning("degirum supported-probe timeout device_type=%s", device_type)
        except Exception as exc:
            parsed = parse_supported_device_types_from_error(exc)
            if parsed:
                for item in parsed:
                    if item not in supported:
                        supported.append(item)
                break
            logger.debug("degirum supported-probe failed device_type=%s error=%s", device_type, exc)
        finally:
            close_method = getattr(model, "close", None)
            if callable(close_method):
                with suppress(Exception):
                    close_method()

    if cache is not None:
        cache[cache_key] = list(supported)
    return supported


def resolve_degirum_runtime_target(
    *,
    logical_selection: object,
    supported_device_types: Sequence[str],
    inference_host_address: str = "@local",
) -> dict[str, Any]:
    logical = normalize_degirum_device_selection(logical_selection)
    supported = [
        normalized
        for item in supported_device_types
        if (normalized := _normalize_supported_device_type(item))
    ]
    supported = list(dict.fromkeys(supported))

    selected = ""
    fallback_used = False
    details: list[str] = []

    def _first_by_kind(kind: str) -> str:
        for dev_type in supported:
            if device_kind_from_type(dev_type) == kind:
                return dev_type
        return ""

    if is_valid_degirum_device_type(logical):
        concrete = _normalize_supported_device_type(logical)
        if concrete in supported:
            selected = concrete
            details.append(f"selected concrete supported type {concrete}")
        else:
            fallback_used = True
            details.append(f"concrete type {logical} not supported")
    elif logical == "gpu":
        selected = _first_by_kind("gpu")
        if not selected:
            fallback_used = True
            details.append("logical GPU unavailable in supported device types")
    elif logical in {"cpu", "auto"}:
        kind = "cpu" if logical == "cpu" else "gpu"
        selected = _first_by_kind(kind)
        if not selected:
            selected = _first_by_kind("cpu")
        if not selected:
            selected = supported[0] if supported else ""
        if logical == "auto":
            details.append("auto resolved using supported list priority")
    else:
        fallback_used = True
        details.append(f"unsupported logical selection {logical}, fallback")

    if not selected:
        cpu_like = _first_by_kind("cpu")
        selected = cpu_like or (supported[0] if supported else "")
        fallback_used = True

    return {
        "logical_selection": logical,
        "final_device_type": selected,
        "inference_host_address": str(inference_host_address),
        "fallback_used": fallback_used,
        "details": "; ".join(details),
        "supported_device_types": supported,
    }


def resolve_effective_degirum_selection(camera_config: dict[str, Any] | None, app_config: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve logical DeGirum selection from camera override + app preferences."""
    cam_cfg = camera_config if isinstance(camera_config, dict) else {}
    app_cfg = app_config if isinstance(app_config, dict) else {}

    override_enabled = bool(cam_cfg.get("degirum_device_override_enabled", False))
    override_value = normalize_degirum_device_selection(
        cam_cfg.get("degirum_device_override", "inherit"),
        allow_inherit=True,
    )
    if override_enabled and override_value != "inherit":
        return {"logical_selection": override_value, "resolution_source": "camera_override"}

    mode = normalize_degirum_device_selection(app_cfg.get("degirum_device_mode", "auto"))
    if mode != "auto":
        return {"logical_selection": mode, "resolution_source": "global_mode"}

    preferred = normalize_degirum_device_selection(app_cfg.get("degirum_preferred_device", "auto"))
    return {"logical_selection": preferred, "resolution_source": "global_preferred"}


def cpu_fallback_candidates_from_supported(supported_device_types: Sequence[str]) -> list[str]:
    supported = [
        normalized
        for item in supported_device_types
        if (normalized := _normalize_supported_device_type(item))
    ]
    cpu_only = [item for item in supported if device_kind_from_type(item) == "cpu"]
    if cpu_only:
        return list(dict.fromkeys(cpu_only))
    if supported:
        return [supported[0]]
    return []


def detect_degirum_devices(
    dg_module: Any,
    *,
    model_name: str = DEFAULT_MODEL,
    zoo_url: str | Path | None = None,
    supported_cache: dict[tuple[str, str], list[str]] | None = None,
) -> list[dict[str, Any]]:
    supported = get_model_supported_device_types(
        dg_module,
        model_name=model_name,
        zoo_url=zoo_url,
        cache=supported_cache,
    )

    records: list[dict[str, Any]] = [
        {
            "id": "auto",
            "label": GUI_LABELS["auto"],
            "kind": "auto",
            "available": bool(supported),
            "details": "Automatyczny wybór na bazie supported device types modelu.",
            "score": 0.0,
            "recommended": False,
            "final_device_type": "",
        },
        {
            "id": "cpu",
            "label": GUI_LABELS["cpu"],
            "kind": "cpu",
            "available": any(device_kind_from_type(item) == "cpu" for item in supported),
            "details": "Mapowanie logiczne CPU -> pierwszy wspierany typ CPU.",
            "score": 0.0,
            "recommended": False,
            "final_device_type": resolve_degirum_runtime_target(
                logical_selection="cpu",
                supported_device_types=supported,
            ).get("final_device_type", ""),
        },
        {
            "id": "gpu",
            "label": GUI_LABELS["gpu"],
            "kind": "gpu",
            "available": any(device_kind_from_type(item) == "gpu" for item in supported),
            "details": "Mapowanie logiczne GPU -> pierwszy wspierany typ GPU.",
            "score": 0.0,
            "recommended": False,
            "final_device_type": resolve_degirum_runtime_target(
                logical_selection="gpu",
                supported_device_types=supported,
            ).get("final_device_type", ""),
        },
    ]

    for item in supported:
        records.append(
            {
                "id": item,
                "label": item,
                "kind": device_kind_from_type(item) or "device_type",
                "available": True,
                "details": f"Wspierany device type modelu: {item}",
                "score": 0.0,
                "recommended": False,
                "final_device_type": item,
            }
        )

    return records


def benchmark_device_candidates(
    dg_module: Any,
    *,
    model_name: str,
    candidates: Sequence[str],
    zoo_url: str | Path | None = None,
    sample_input: Any = None,
    inference_runs: int = 2,
    config: dict[str, Any] | None = None,
    supported_cache: dict[tuple[str, str], list[str]] | None = None,
) -> dict[str, Any]:
    normalized_model = coerce_optional_str(model_name) or ""
    normalized_zoo = coerce_optional_str(zoo_url if zoo_url is not None else MODELS_PATH / normalized_model) or ""
    supported = get_model_supported_device_types(
        dg_module,
        model_name=normalized_model,
        zoo_url=normalized_zoo,
        cache=supported_cache,
    )

    logical_candidates = [normalize_degirum_device_selection(item) for item in candidates]
    logical_candidates = [item for item in logical_candidates if item not in {"inherit"}]
    logical_candidates = list(dict.fromkeys(logical_candidates or ["auto", "cpu"]))

    rows: list[dict[str, Any]] = []
    for logical in logical_candidates:
        resolved = resolve_degirum_runtime_target(
            logical_selection=logical,
            supported_device_types=supported,
        )
        final_device_type = str(resolved.get("final_device_type") or "")
        if not final_device_type:
            rows.append(
                {
                    "device": logical,
                    "kind": logical,
                    "available": False,
                    "stable": False,
                    "load_time_ms": None,
                    "inference_time_ms": [],
                    "mean_inference_time_ms": None,
                    "score": 0.0,
                    "error": "Brak wspieranego final_device_type dla modelu.",
                    "final_device_type": "",
                }
            )
            continue

        started = time.perf_counter()
        model = None
        try:
            raw_kwargs = {
                "model_name": normalized_model,
                "inference_host_address": "@local",
                "zoo_url": normalized_zoo,
                "device_type": final_device_type,
            }
            load_kwargs = sanitize_degirum_load_model_kwargs(raw_kwargs)
            _log_load_model_diagnostics(f"degirum benchmark {logical}", raw_kwargs, load_kwargs)
            logger.debug("degirum benchmark load attempt logical=%s final_device_type=%s", logical, final_device_type)
            model = dg_module.load_model(**load_kwargs)
            logger.debug("degirum benchmark load success logical=%s final_device_type=%s", logical, final_device_type)
            load_ms = (time.perf_counter() - started) * 1000.0
            infer_ms_values: list[float] = []
            infer_fn = getattr(model, "predict", None)
            if callable(infer_fn):
                for _ in range(max(1, min(3, int(inference_runs)))):
                    infer_start = time.perf_counter()
                    infer_fn(sample_input)
                    infer_ms_values.append((time.perf_counter() - infer_start) * 1000.0)
            mean_ms = sum(infer_ms_values) / len(infer_ms_values) if infer_ms_values else None
            rows.append(
                {
                    "device": logical,
                    "kind": logical,
                    "available": True,
                    "stable": True,
                    "load_time_ms": round(load_ms, 3),
                    "inference_time_ms": [round(v, 3) for v in infer_ms_values],
                    "mean_inference_time_ms": round(mean_ms, 3) if mean_ms is not None else None,
                    "score": round(1_000_000.0 / (1.0 + load_ms + (mean_ms or 0.0)), 3),
                    "error": None,
                    "final_device_type": final_device_type,
                }
            )
        except Exception as exc:
            logger.debug(
                "degirum benchmark load failure logical=%s final_device_type=%s error=%s",
                logical,
                final_device_type,
                exc,
            )
            rows.append(
                {
                    "device": logical,
                    "kind": logical,
                    "available": False,
                    "stable": False,
                    "load_time_ms": None,
                    "inference_time_ms": [],
                    "mean_inference_time_ms": None,
                    "score": 0.0,
                    "error": str(exc),
                    "final_device_type": final_device_type,
                }
            )
        finally:
            close_method = getattr(model, "close", None)
            if callable(close_method):
                try:
                    close_method()
                except Exception:
                    pass

    rows.sort(key=lambda item: (item.get("available"), item.get("stable"), item.get("score", 0.0)), reverse=True)
    available_devices = [row["device"] for row in rows if row.get("available")]
    payload = {
        "model_name": normalized_model,
        "supported_device_types": supported,
        "results": rows,
        "available_devices": available_devices,
    }
    if config is not None:
        config["degirum_last_benchmark"] = payload
        config["degirum_available_devices"] = available_devices
    return payload


def choose_best_degirum_device(
    dg_module: Any,
    *,
    model_name: str,
    candidates: Sequence[str],
    zoo_url: str | Path | None = None,
    sample_input: Any = None,
    inference_runs: int = 2,
    config: dict[str, Any] | None = None,
    auto_select: bool = False,
    supported_cache: dict[tuple[str, str], list[str]] | None = None,
) -> str:
    benchmark = benchmark_device_candidates(
        dg_module,
        model_name=model_name,
        candidates=candidates,
        zoo_url=zoo_url,
        sample_input=sample_input,
        inference_runs=inference_runs,
        config=config,
        supported_cache=supported_cache,
    )
    rows = benchmark.get("results", [])
    selected = "cpu"
    if rows:
        selected = str(rows[0].get("device") or "cpu")
    if config is not None and auto_select:
        config["degirum_preferred_device"] = selected
    return selected


__all__ = [
    "benchmark_device_candidates",
    "build_degirum_load_model_kwargs",
    "choose_best_degirum_device",
    "coerce_optional_str",
    "coerce_pathlike_to_str",
    "cpu_fallback_candidates_from_supported",
    "detect_degirum_devices",
    "device_kind_from_type",
    "get_model_supported_device_types",
    "load_model_with_timeout",
    "parse_supported_device_types_from_error",
    "resolve_effective_degirum_selection",
    "resolve_degirum_runtime_target",
    "sanitize_degirum_load_model_kwargs",
]
