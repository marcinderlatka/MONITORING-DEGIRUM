from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _CpuSnapshot:
    total: int
    idle: int


@dataclass
class _IoSnapshot:
    read_sectors: int
    write_sectors: int
    ts: float


@dataclass
class SystemMetricsSampler:
    """Lightweight Linux system metrics sampler based on /proc files."""

    _cpu_prev_total: _CpuSnapshot | None = None
    _cpu_prev_cores: list[_CpuSnapshot] = field(default_factory=list)
    _io_prev: _IoSnapshot | None = None

    def collect(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "cpu_total_percent": None,
            "cpu_per_core_percent": [],
            "memory_used_mb": None,
            "memory_available_mb": None,
            "memory_total_mb": None,
            "memory_used_percent": None,
            "swap_used_mb": None,
            "swap_free_mb": None,
            "swap_total_mb": None,
            "swap_used_percent": None,
            "load_average": None,
            "io_read_mbps": None,
            "io_write_mbps": None,
            "timestamp": time.time(),
        }

        meminfo = _read_meminfo()
        if meminfo:
            mem_total_kb = meminfo.get("MemTotal", 0)
            mem_available_kb = meminfo.get("MemAvailable", 0)
            mem_used_kb = max(0, mem_total_kb - mem_available_kb)
            metrics["memory_total_mb"] = mem_total_kb / 1024.0
            metrics["memory_available_mb"] = mem_available_kb / 1024.0
            metrics["memory_used_mb"] = mem_used_kb / 1024.0
            if mem_total_kb > 0:
                metrics["memory_used_percent"] = (mem_used_kb / mem_total_kb) * 100.0

            swap_total_kb = meminfo.get("SwapTotal", 0)
            swap_free_kb = meminfo.get("SwapFree", 0)
            swap_used_kb = max(0, swap_total_kb - swap_free_kb)
            metrics["swap_total_mb"] = swap_total_kb / 1024.0
            metrics["swap_free_mb"] = swap_free_kb / 1024.0
            metrics["swap_used_mb"] = swap_used_kb / 1024.0
            if swap_total_kb > 0:
                metrics["swap_used_percent"] = (swap_used_kb / swap_total_kb) * 100.0
            else:
                metrics["swap_used_percent"] = 0.0

        cpu_total, cpu_per_core = self._collect_cpu_percent()
        metrics["cpu_total_percent"] = cpu_total
        metrics["cpu_per_core_percent"] = cpu_per_core

        with _suppress_exceptions():
            one, five, fifteen = os.getloadavg()
            metrics["load_average"] = (float(one), float(five), float(fifteen))

        io_read_mbps, io_write_mbps = self._collect_io_mbps()
        metrics["io_read_mbps"] = io_read_mbps
        metrics["io_write_mbps"] = io_write_mbps

        return metrics

    def _collect_cpu_percent(self) -> tuple[float | None, list[float | None]]:
        total_raw, core_raw = _read_cpu_times()
        if total_raw is None:
            return None, []

        total_prev = self._cpu_prev_total
        self._cpu_prev_total = total_raw

        per_core: list[float | None] = []
        if not self._cpu_prev_cores:
            self._cpu_prev_cores = core_raw
        else:
            for idx, curr in enumerate(core_raw):
                prev = self._cpu_prev_cores[idx] if idx < len(self._cpu_prev_cores) else None
                per_core.append(_cpu_percent(prev, curr))
            self._cpu_prev_cores = core_raw

        total_percent = _cpu_percent(total_prev, total_raw)
        return total_percent, per_core

    def _collect_io_mbps(self) -> tuple[float | None, float | None]:
        stats = _read_diskstats_sectors()
        if stats is None:
            return None, None
        now = time.time()
        snapshot = _IoSnapshot(read_sectors=stats[0], write_sectors=stats[1], ts=now)
        prev = self._io_prev
        self._io_prev = snapshot
        if prev is None:
            return None, None
        elapsed = max(1e-6, now - prev.ts)
        read_bps = ((snapshot.read_sectors - prev.read_sectors) * 512.0) / elapsed
        write_bps = ((snapshot.write_sectors - prev.write_sectors) * 512.0) / elapsed
        return read_bps / (1024.0 * 1024.0), write_bps / (1024.0 * 1024.0)


class _suppress_exceptions:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return True


def _read_meminfo() -> dict[str, int]:
    info: dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                key, _, rest = line.partition(":")
                val = rest.strip().split()[0]
                if val.isdigit():
                    info[key] = int(val)
    except OSError:
        return {}
    return info


def _read_cpu_times() -> tuple[_CpuSnapshot | None, list[_CpuSnapshot]]:
    try:
        with open("/proc/stat", "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return None, []

    total: _CpuSnapshot | None = None
    cores: list[_CpuSnapshot] = []
    for line in lines:
        if not line.startswith("cpu"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        values = [int(v) for v in parts[1:] if v.isdigit()]
        if len(values) < 4:
            continue
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        snap = _CpuSnapshot(total=sum(values), idle=idle)
        if parts[0] == "cpu":
            total = snap
        elif parts[0].startswith("cpu"):
            cores.append(snap)
    return total, cores


def _cpu_percent(prev: _CpuSnapshot | None, curr: _CpuSnapshot) -> float | None:
    if prev is None:
        return None
    total_delta = curr.total - prev.total
    idle_delta = curr.idle - prev.idle
    if total_delta <= 0:
        return 0.0
    usage = (1.0 - (idle_delta / total_delta)) * 100.0
    return max(0.0, min(100.0, usage))


def _read_diskstats_sectors() -> tuple[int, int] | None:
    try:
        with open("/proc/diskstats", "r", encoding="utf-8") as f:
            read_sectors = 0
            write_sectors = 0
            for line in f:
                parts = line.split()
                if len(parts) < 14:
                    continue
                dev = parts[2]
                if dev.startswith("loop") or dev.startswith("ram"):
                    continue
                if dev.endswith(tuple(str(i) for i in range(10))):
                    # Skip partitions to avoid double counting where possible.
                    continue
                read_sectors += int(parts[5])
                write_sectors += int(parts[9])
            return read_sectors, write_sectors
    except OSError:
        return None
