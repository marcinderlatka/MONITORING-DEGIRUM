"""Runtime helper utilities that are test-friendly and dependency-light."""


def classify_camera_setting_changes(old_camera: dict, new_camera: dict, restart_required_fields: set[str]) -> tuple[list[str], list[str]]:
    """Return changed camera keys and subset that require worker restart."""
    changed_keys = sorted(key for key in (set(old_camera) | set(new_camera)) if old_camera.get(key) != new_camera.get(key))
    restart_keys = [key for key in changed_keys if key in restart_required_fields]
    return changed_keys, restart_keys


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
