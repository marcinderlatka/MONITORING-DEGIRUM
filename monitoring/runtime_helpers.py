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


def compute_letterboxed_rect(frame_width: int, frame_height: int, canvas_width: int, canvas_height: int) -> tuple[int, int, int, int]:
    """Compute visible image rectangle for a frame letterboxed into a canvas."""
    if frame_width <= 0 or frame_height <= 0 or canvas_width <= 0 or canvas_height <= 0:
        return 0, 0, max(1, canvas_width), max(1, canvas_height)
    scale = min(canvas_width / frame_width, canvas_height / frame_height)
    new_w = max(1, int(frame_width * scale))
    new_h = max(1, int(frame_height * scale))
    x0 = (canvas_width - new_w) // 2
    y0 = (canvas_height - new_h) // 2
    return x0, y0, new_w, new_h


def camera_overlay_anchor(image_rect: tuple[int, int, int, int], box_size: tuple[int, int], padding: int = 10) -> tuple[int, int]:
    """Anchor bottom-left HUD box inside the visible image rectangle."""
    x0, y0, _iw, ih = image_rect
    bw, bh = box_size
    x = x0 + padding
    y = y0 + ih - bh - padding
    return max(x0, x), max(y0, y)


def thumbnail_load_outcome(image: object) -> str:
    """Classify thumbnail result to avoid indefinite loading state."""
    if image is None:
        return "fallback"
    is_null = getattr(image, "isNull", None)
    if callable(is_null):
        try:
            return "fallback" if bool(is_null()) else "success"
        except Exception:
            return "fallback"
    return "success"
