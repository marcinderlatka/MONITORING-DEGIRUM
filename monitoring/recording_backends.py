"""Recording backend abstractions used by :mod:`monitoring.workers`."""

from __future__ import annotations

import logging
import shlex
import subprocess
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
        preset: str = "veryfast",
        tune: str = "zerolatency",
        crf: int | None = 23,
        movflags: str = "+faststart",
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
        self._process: subprocess.Popen[bytes] | None = None
        self._ffmpeg_exit_code: int | None = None
        self._stderr_summary = ""
        self._command_line = ""

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
            self.codec,
            "-preset",
            self.preset,
            "-tune",
            self.tune,
        ]
        if self.crf is not None:
            cmd.extend(["-crf", str(int(self.crf))])
        cmd.extend([
            "-movflags",
            self.movflags,
            "-pix_fmt",
            "yuv420p",
            self.filepath,
        ])
        self._command_line = " ".join(shlex.quote(part) for part in cmd)
        logger.info("starting ffmpeg backend: %s", self._command_line)
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
        stderr_text = ""
        with suppress(Exception):
            if self._process.stderr is not None:
                stderr_text = self._process.stderr.read().decode("utf-8", errors="replace")
        self._ffmpeg_exit_code = self._process.wait(timeout=5.0)
        if stderr_text:
            lines = [ln.strip() for ln in stderr_text.splitlines() if ln.strip()]
            self._stderr_summary = " | ".join(lines[-6:])[:1500]
        self._process = None

    @property
    def ffmpeg_exit_code(self) -> int | None:
        return self._ffmpeg_exit_code

    @property
    def stderr_summary(self) -> str:
        return self._stderr_summary
