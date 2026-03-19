"""Recording backend abstractions used by :mod:`monitoring.workers`."""

from __future__ import annotations

import logging
import shlex
import subprocess
import time
from abc import ABC, abstractmethod
from contextlib import suppress
from shutil import which

import degirum_tools  # type: ignore
import numpy as np

logger = logging.getLogger(__name__)


class BaseRecordingBackend(ABC):
    """Common interface for recording implementations."""

    backend_name = "base"

    @abstractmethod
    def open(self) -> None:
        """Allocate writer resources."""

    @abstractmethod
    def write(self, frame: np.ndarray) -> None:
        """Write a single video frame."""

    @abstractmethod
    def close(self) -> None:
        """Release resources and flush writer state."""

    @property
    def ffmpeg_exit_code(self) -> int | None:
        return None

    @property
    def stderr_summary(self) -> str:
        return ""


class DeGirumWriterBackend(BaseRecordingBackend):
    """Adapter for the existing ``degirum_tools.VideoWriter`` path."""

    backend_name = "current"

    def __init__(self, filepath: str, width: int, height: int, fps: float) -> None:
        self.filepath = filepath
        self.width = int(width)
        self.height = int(height)
        self.fps = float(max(1.0, fps))
        self._writer = None

    def open(self) -> None:
        self._writer = degirum_tools.VideoWriter(self.filepath, self.width, self.height, self.fps)

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            raise RuntimeError("backend not opened")
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            with suppress(AttributeError):
                self._writer.release()
            self._writer = None


class FFmpegPipeBackend(BaseRecordingBackend):
    """Minimal rawvideo pipe backend used for ffmpeg based writing."""

    backend_name = "ffmpeg"

    def __init__(
        self,
        filepath: str,
        width: int,
        height: int,
        fps: float,
        *,
        codec: str = "libx264",
        preset: str = "superfast",
        tune: str = "zerolatency",
        crf: int | None = 28,
        movflags: str = "+faststart",
        profile: str = "latency",
    ) -> None:
        self.filepath = filepath
        self.width = int(width)
        self.height = int(height)
        self.fps = float(max(1.0, fps))
        self.codec = str(codec or "libx264")
        self.preset = str(preset or "veryfast")
        self.tune = str(tune or "zerolatency")
        self.crf = None if crf is None else int(crf)
        self.movflags = str(movflags or "+faststart")
        self.profile = str(profile or "latency").strip().lower()
        self._process: subprocess.Popen[bytes] | None = None
        self._ffmpeg_exit_code: int | None = None
        self._stderr_summary = ""
        self._stderr_tail = ""
        self._command_line = ""
        self._broken_pipe = False
        self._broken_pipe_count = 0
        self._write_retry_count = 0
        self._frames_written = 0
        self._write_total_ms = 0.0
        self._close_timeout = False

    def open(self) -> None:
        ffmpeg_bin = which("ffmpeg") or "ffmpeg"
        active_profile = self.profile if self.profile in {"latency", "throughput"} else "latency"
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            f"{self.fps:g}",
            "-thread_queue_size",
            "1024",
            "-i",
            "-",
            "-an",
            "-vcodec",
            self.codec,
            "-preset",
            self.preset,
            "-tune",
            self.tune,
        ]
        if self.crf is not None:
            cmd.extend(["-crf", str(int(self.crf))])
        if active_profile == "latency":
            cmd.extend(["-fflags", "nobuffer"])
        cmd.extend([
            "-movflags",
            self.movflags,
            "-pix_fmt",
            "yuv420p",
            self.filepath,
        ])
        if active_profile == "latency":
            cmd[-1:-1] = ["-flush_packets", "1"]
        self._command_line = " ".join(shlex.quote(part) for part in cmd)
        logger.info(
            "starting ffmpeg backend: %s (codec=%s preset=%s tune=%s crf=%s movflags=%s profile=%s)",
            self._command_line,
            self.codec,
            self.preset,
            self.tune,
            self.crf if self.crf is not None else "none",
            self.movflags,
            active_profile,
        )
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=self.width * self.height * 3 * (8 if active_profile == "throughput" else 2),
        )

    def write(self, frame: np.ndarray) -> None:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("ffmpeg backend not opened")
        if self._process.poll() is not None:
            raise RuntimeError(f"ffmpeg terminated early with code {self._process.returncode}")
        if bool(frame.flags["C_CONTIGUOUS"]):
            payload = memoryview(frame).cast("B")
        else:
            payload = memoryview(np.ascontiguousarray(frame)).cast("B")
        start = time.perf_counter()
        for attempt in range(2):
            try:
                self._process.stdin.write(payload)
                self._frames_written += 1
                self._write_total_ms += (time.perf_counter() - start) * 1000.0
                return
            except BrokenPipeError as exc:
                self._broken_pipe = True
                self._broken_pipe_count += 1
                if attempt == 0:
                    self._write_retry_count += 1
                    continue
                raise RuntimeError(f"ffmpeg pipe write failed: {exc}") from exc
            except OSError as exc:
                if attempt == 0:
                    self._write_retry_count += 1
                    continue
                raise RuntimeError(f"ffmpeg pipe write failed: {exc}") from exc

    def close(self) -> None:
        if self._process is None:
            return
        if self._process.stdin is not None:
            with suppress(BrokenPipeError, OSError):
                self._process.stdin.close()
        stderr_text = ""
        with suppress(Exception):
            if self._process.stderr is not None:
                stderr_text = self._process.stderr.read().decode("utf-8", errors="replace")
        try:
            self._ffmpeg_exit_code = self._process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self._close_timeout = True
            self._process.kill()
            self._ffmpeg_exit_code = self._process.wait(timeout=2.0)
        if stderr_text:
            lines = [ln.strip() for ln in stderr_text.splitlines() if ln.strip()]
            self._stderr_tail = " | ".join(lines[-20:])[:3000]
            self._stderr_summary = " | ".join(lines[-6:])[:1500]
        write_avg_ms = self._write_total_ms / self._frames_written if self._frames_written > 0 else 0.0
        logger.info(
            "ffmpeg backend closed: cmd=%s exit_code=%s broken_pipe=%s timeout=%s write_avg_ms=%.3f retry_count=%s broken_pipe_count=%s stderr=%s",
            self._command_line,
            self._ffmpeg_exit_code,
            self._broken_pipe,
            self._close_timeout,
            write_avg_ms,
            self._write_retry_count,
            self._broken_pipe_count,
            (self._stderr_summary[:400] if self._stderr_summary else ""),
        )
        if self._broken_pipe:
            logger.warning("ffmpeg broken pipe detected: cmd=%s", self._command_line)
        if self._close_timeout:
            logger.warning("ffmpeg close timeout detected: cmd=%s", self._command_line)
        self._process = None

    @property
    def ffmpeg_exit_code(self) -> int | None:
        return self._ffmpeg_exit_code

    @property
    def stderr_summary(self) -> str:
        return self._stderr_summary
