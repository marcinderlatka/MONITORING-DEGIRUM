from monitoring.runtime_helpers import compute_effective_writer_fps


def test_compute_effective_writer_fps_prefers_rtsp_limit():
    assert compute_effective_writer_fps(rtsp_fps=5, measured_fps=20.0, stream_fps=25.0) == 5.0


def test_compute_effective_writer_fps_uses_measured_when_unthrottled():
    assert compute_effective_writer_fps(rtsp_fps=0, measured_fps=7.5, stream_fps=25.0) == 7.5


def test_compute_effective_writer_fps_falls_back_to_stream():
    assert compute_effective_writer_fps(rtsp_fps=0, measured_fps=0.0, stream_fps=30.0) == 30.0
