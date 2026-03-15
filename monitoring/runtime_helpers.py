"""Runtime helper utilities that are test-friendly and dependency-light."""


def compute_effective_writer_fps(rtsp_fps: int, detect_fps: float, stream_fps: float) -> float:
    """Compute MP4 writer FPS so playback matches processed frame cadence."""
    if rtsp_fps > 0:
        return float(max(1.0, rtsp_fps))
    if detect_fps > 0:
        if stream_fps > 0:
            return float(max(1.0, min(stream_fps, detect_fps)))
        return float(max(1.0, detect_fps))
    if stream_fps > 0:
        return float(max(1.0, stream_fps))
    return 1.0
