"""Recording backend abstractions used by :mod:`monitoring.workers`."""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from contextlib import suppress
from shutil import which

import degirum_tools  # type: ignore
import numpy as np


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

    def __init__(self, filepath: str, width: int, height: int, fps: float) -> None:
        self.filepath = filepath
        self.width = int(width)
        self.height = int(height)
        self.fps = float(max(1.0, fps))
        self._process: subprocess.Popen[bytes] | None = None
        self._ffmpeg_exit_code: int | None = None

    def open(self) -> None:
        ffmpeg_bin = which("ffmpeg") or "ffmpeg"
        cmd = [
            ffmpeg_bin,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            f"{self.fps:g}",
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            self.filepath,
        ]
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("ffmpeg backend not opened")
        if self._process.poll() is not None:
            raise RuntimeError(f"ffmpeg terminated early with code {self._process.returncode}")
        self._process.stdin.write(frame.tobytes())

    def close(self) -> None:
        if self._process is None:
            return
        if self._process.stdin is not None:
            with suppress(BrokenPipeError, OSError):
                self._process.stdin.close()
        self._ffmpeg_exit_code = self._process.wait(timeout=5.0)
        self._process = None

    @property
    def ffmpeg_exit_code(self) -> int | None:
        return self._ffmpeg_exit_code
