"""Runtime helper utilities that are test-friendly and dependency-light."""


def compute_effective_writer_fps(rtsp_fps: int, measured_fps: float, stream_fps: float) -> float:
    """Compute MP4 writer FPS so playback matches written frame cadence."""
    if rtsp_fps > 0:
        return float(max(1, rtsp_fps))
    if measured_fps > 0.1:
        return float(max(1.0, measured_fps))
    if stream_fps > 0.1:
        return float(max(1.0, stream_fps))
    return 5.0
