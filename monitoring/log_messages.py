from __future__ import annotations

import re
from typing import Any

LOG_MESSAGES: dict[str, str] = {
    "performance_summary_title": "Podsumowanie wydajności",
    "details_prefix": "Szczegóły",
    "details_technical_prefix": "Szczegóły techniczne",
    "show_more": "Pokaż więcej",
    "show_less": "Pokaż mniej",
    "camera_label": "Kamera",
    "detection_label": "Detekcja",
    "recording_started": "Nagrywanie rozpoczęte",
    "recording_finished": "Nagrywanie zakończone",
    "detection_started": "Detekcja rozpoczęta",
    "detection_finished": "Detekcja zakończona",
    "worker_metrics_summary_action": "Podsumowanie metryk workera",
    "ui_worker_metrics_summary_action": "Podsumowanie metryk UI + worker",
}

PERFORMANCE_PARAM_LABELS: dict[str, str] = {
    "mode": "Tryb podglądu",
    "overload": "Tryb przeciążenia",
    "capture_fps": "FPS przechwytywania",
    "infer_fps": "FPS detekcji",
    "preview_emit_fps": "FPS emisji podglądu",
    "ui_render_ms": "Render UI [ms]",
    "thumb_ms": "Miniatura [ms]",
    "grid_ms": "Siatka [ms]",
    "grid_avg_ms": "Średni czas siatki [ms]",
    "grid_fps": "FPS siatki",
    "main_ms": "Widok główny [ms]",
    "queue_size": "Rozmiar kolejki",
    "dropped_frames": "Utracone klatki",
    "cpu_percent": "CPU [%]",
    "rss_mb": "Pamięć RSS [MB]",
    "fp_proxy": "Współczynnik FP (proxy)",
    "avg_conf": "Średnia pewność",
    "trigger_h": "Wyzwolenia / godz.",
}


def msg(key: str, **kwargs: Any) -> str:
    text = LOG_MESSAGES.get(key, key)
    return text.format(**kwargs) if kwargs else text


def format_performance_summary(params: dict[str, Any]) -> str:
    lines = [msg("performance_summary_title")]
    for key, value in params.items():
        label = PERFORMANCE_PARAM_LABELS.get(key, key.replace("_", " "))
        lines.append(f"{label}: {value}")
    return "\n".join(lines)


def summarize_performance_details(details: str) -> str:
    params: dict[str, str] = {}
    for key, value in re.findall(r"([a-zA-Z0-9_]+)=([^\\s]+)", details or ""):
        params[key] = value
    if not params:
        return details
    return format_performance_summary(params)
