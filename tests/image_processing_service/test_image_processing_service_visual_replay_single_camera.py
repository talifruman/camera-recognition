"""Integration tests for the IPS single-camera visual replay runner."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import pytest

_SRC_DIR = Path(__file__).resolve().parents[2] / "src"
_TESTS_DIR = Path(__file__).resolve().parents[1]
for _path in (_SRC_DIR, _TESTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from frame_ingestion_gateway.replay_helpers import get_asset_path  # type: ignore[import-not-found]
from image_processing_service.visual_replay_single_camera import (  # type: ignore[import-not-found]
    DEFAULT_INPUT_VIDEO_PATH,
    EXISTING_INPUT_VIDEO_PATH,
    ReplayFrameRecord,
    RUNTIME_TRACE_PER_FRAME_NAME,
    VisualReplayError,
    _resolve_object_detection_is_real,
    _build_runtime_trace_per_frame_payload,
    _build_runtime_trace_summary,
    run_visual_replay,
)

_FPS = 5.0
_FRAME_COUNT = 2


def test_object_detection_mode_uses_real_implementation_type(tmp_path: Path) -> None:
    """Resolve object detection mode as real when YAML implementation_type is real."""
    od_config_path = tmp_path / "object_detection_real.yaml"
    od_config_path.write_text(
        '\n'.join(
            [
                'implementation_type: "real"',
                'model_path: "models/object_detection/yolo11s.pt"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rpm_config = {
        "models": {
            "object_detection_config": str(od_config_path),
        }
    }

    assert _resolve_object_detection_is_real(rpm_config) is True


def test_object_detection_mode_uses_stub_implementation_type(tmp_path: Path) -> None:
    """Resolve object detection mode as stub when YAML implementation_type is stub."""
    od_config_path = tmp_path / "object_detection_stub.yaml"
    od_config_path.write_text(
        '\n'.join(
            [
                'implementation_type: "stub"',
                'model_path: "models/object_detection/yolo11s.pt"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rpm_config = {
        "models": {
            "object_detection_config": str(od_config_path),
        }
    }

    assert _resolve_object_detection_is_real(rpm_config) is False


def _write_temp_video(video_path: Path) -> None:
    """Create a short mp4 from existing gateway replay assets."""
    frame_paths = [
        get_asset_path("frame_000008.jpg"),
        get_asset_path("frame_000009.jpg"),
    ]
    frames = []
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Unable to decode frame asset: {frame_path}")
        frames.append(frame)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]
        _FPS,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open temporary video writer: {video_path}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


@pytest.mark.integration
def test_visual_replay_fails_clearly_when_default_video_is_missing() -> None:
    """Fail with the actionable message when default and fallback inputs are missing."""
    missing_default = DEFAULT_INPUT_VIDEO_PATH.parent / "missing_default.mp4"
    missing_existing = EXISTING_INPUT_VIDEO_PATH.parent / "missing_existing.mp4"
    original_default = DEFAULT_INPUT_VIDEO_PATH
    original_existing = EXISTING_INPUT_VIDEO_PATH
    module = sys.modules[
        "image_processing_service.visual_replay_single_camera"
    ]
    setattr(module, "DEFAULT_INPUT_VIDEO_PATH", missing_default)
    setattr(module, "EXISTING_INPUT_VIDEO_PATH", missing_existing)
    try:
        with pytest.raises(VisualReplayError) as exc_info:
            run_visual_replay(input_video_path=None)
    finally:
        setattr(module, "DEFAULT_INPUT_VIDEO_PATH", original_default)
        setattr(module, "EXISTING_INPUT_VIDEO_PATH", original_existing)
    message = str(exc_info.value)
    assert str(missing_default) in message
    assert str(missing_existing) in message
    assert "Expected one of these paths" in message


@pytest.mark.integration
def test_visual_replay_generates_artifacts_and_runtime_metrics(
    tmp_path: Path,
) -> None:
    """Run the real gateway->IPS->RPM replay path against a temporary mp4."""
    video_path = tmp_path / "single_camera_test.mp4"
    output_dir = tmp_path / "output"
    _write_temp_video(video_path)

    result = run_visual_replay(
        input_video_path=video_path,
        output_dir=output_dir,
        replay_fps=_FPS,
    )

    assert result.output_overlay_path.exists()
    assert result.summary_path.exists()
    assert result.metrics_csv_path.exists()
    runtime_trace_path = output_dir / "runtime_trace.csv"
    runtime_trace_summary_path = output_dir / "runtime_trace_summary.json"
    runtime_trace_per_frame_path = output_dir / RUNTIME_TRACE_PER_FRAME_NAME
    assert runtime_trace_path.exists()
    assert runtime_trace_summary_path.exists()
    assert runtime_trace_per_frame_path.exists()
    assert result.frames_read == _FRAME_COUNT
    assert result.frames_submitted == _FRAME_COUNT
    assert result.frames_processed >= 1
    assert result.frames_rendered == _FRAME_COUNT
    assert result.gateway_frames_in_total >= 1
    assert result.gateway_frames_published_total >= 1
    assert result.ips_processed_total >= 1
    assert result.ftl_calls_last_frame >= 1
    assert result.final_service_state in {"STOPPED", "DEGRADED"}
    assert result.final_active_workers == 0

    with open(result.metrics_csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == _FRAME_COUNT
    assert {row["camera_id"] for row in rows} == {"camera_1"}

    with open(runtime_trace_path, newline="", encoding="utf-8") as handle:
        trace_rows = list(csv.DictReader(handle))
    assert len(trace_rows) == _FRAME_COUNT
    assert "motion_stage_ran" in trace_rows[0]
    assert "stop_reason" in trace_rows[0]
    assert "object_detection_total_ms" in trace_rows[0]
    assert "face_detection_total_ms" in trace_rows[0]
    assert "face_recognition_total_ms" in trace_rows[0]

    with open(runtime_trace_per_frame_path, encoding="utf-8") as handle:
        per_frame_payload = json.load(handle)
    assert "frames" in per_frame_payload
    assert len(per_frame_payload["frames"]) == _FRAME_COUNT
    first_frame = per_frame_payload["frames"][0]
    assert "motion_detection" in first_frame
    assert "object_detection" in first_frame
    assert "face_detection" in first_frame
    assert "face_recognition" in first_frame
    assert "totals_for_frame" in first_frame


@pytest.mark.integration
def test_visual_replay_per_stage_timing_is_real(tmp_path: Path) -> None:
    """Persist non-placeholder stage timing for frames where motion stage ran."""
    video_path = tmp_path / "single_camera_test.mp4"
    output_dir = tmp_path / "output"
    _write_temp_video(video_path)

    run_visual_replay(
        input_video_path=video_path,
        output_dir=output_dir,
        replay_fps=_FPS,
    )

    runtime_trace_path = output_dir / "runtime_trace.csv"
    with open(runtime_trace_path, newline="", encoding="utf-8") as handle:
        trace_rows = list(csv.DictReader(handle))
    motion_rows = [row for row in trace_rows if row.get("motion_stage_ran") == "True"]
    if motion_rows:
        assert any(float(row.get("motion_detection_ms", "0") or 0.0) > 0.0 for row in motion_rows)
        return
    processed_rows = [row for row in trace_rows if row.get("ips_processed") == "True"]
    assert processed_rows
    assert any(float(row.get("rpm_total_ms", "0") or 0.0) > 0.0 for row in processed_rows)


@pytest.mark.integration
def test_visual_replay_gate_ordering_verified(tmp_path: Path) -> None:
    """Verify gate ordering diagnostics report no stage ordering violations."""
    video_path = tmp_path / "single_camera_test.mp4"
    output_dir = tmp_path / "output"
    _write_temp_video(video_path)

    run_visual_replay(
        input_video_path=video_path,
        output_dir=output_dir,
        replay_fps=_FPS,
    )

    summary_path = output_dir / "runtime_trace_summary.json"
    with open(summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)
    assert int(summary.get("gate_violation_count", 0)) == 0

    runtime_trace_path = output_dir / "runtime_trace.csv"
    with open(runtime_trace_path, newline="", encoding="utf-8") as handle:
        trace_rows = list(csv.DictReader(handle))
    assert all(not (row.get("gate_violations") or "").strip() for row in trace_rows)


@pytest.mark.integration
def test_visual_replay_bottleneck_summary_fields(tmp_path: Path) -> None:
    """Write runtime summary with throughput and bottleneck stage estimates."""
    video_path = tmp_path / "single_camera_test.mp4"
    output_dir = tmp_path / "output"
    _write_temp_video(video_path)

    run_visual_replay(
        input_video_path=video_path,
        output_dir=output_dir,
        replay_fps=_FPS,
    )

    summary_path = output_dir / "runtime_trace_summary.json"
    with open(summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)
    assert "estimated_input_fps" in summary
    assert "estimated_rpm_fps" in summary
    assert "throughput_ratio" in summary
    assert summary.get("bottleneck_stage_guess", "unknown") != "unknown"


def test_runtime_trace_summary_reports_warmup_backend_and_call_scoped_averages() -> None:
    """Summaries must separate warmup, backend metadata, and call-scoped stage averages."""
    warmup_record = ReplayFrameRecord(
        frame_index=0,
        frame_id="warmup-0",
        timestamp_ms=0,
        original_frame=cv2.imread(str(get_asset_path("frame_000008.jpg")), cv2.IMREAD_COLOR),
        submitted_monotonic_s=0.0,
    )
    measured_record = ReplayFrameRecord(
        frame_index=1,
        frame_id="measured-1",
        timestamp_ms=33,
        original_frame=cv2.imread(str(get_asset_path("frame_000009.jpg")), cv2.IMREAD_COLOR),
        submitted_monotonic_s=0.033,
    )

    assert warmup_record.original_frame is not None
    assert measured_record.original_frame is not None

    warmup_record.ips_processed = True
    warmup_record.trace_status = "processed"
    warmup_record.rpm_total_ms = 120.0
    warmup_record.queue_wait_ms = 80.0
    warmup_record.object_detection_total_ms = 30.0
    warmup_record.object_detection_call_count = 1
    warmup_record.face_detection_total_ms = 14.0
    warmup_record.face_detection_call_count = 1
    warmup_record.face_recognition_total_ms = 7.0
    warmup_record.face_recognition_call_count = 1
    warmup_record.persons_count = 4
    warmup_record.person_rois_count = 4
    warmup_record.object_detection_model_path = "models/object_detection/yolo11m.pt"
    warmup_record.object_detection_model_name = "yolo11m.pt"
    warmup_record.object_detection_device_provider = "ultralytics:cpu"
    warmup_record.object_detection_input_size = "640x640"
    warmup_record.object_detection_confidence_threshold = "0.25"
    warmup_record.object_detection_nms_threshold = "0.45"
    warmup_record.rpm_frame_metrics = {
        "is_warmup": True,
        "face_detection_backend": "onnxruntime",
        "face_detection_device_provider": "CPUExecutionProvider",
        "face_detection_model_path": "models/face_detection/det_500m.onnx",
        "face_detection_input_size": "640x640",
        "face_detection_confidence_threshold": "0.5",
        "face_recognition_backend": "onnxruntime",
        "face_recognition_device_provider": "CPUExecutionProvider",
        "face_recognition_model_path": "models/face_recognition/w600k_mbf.onnx",
        "face_recognition_embedding_model": "w600k_mbf.onnx",
        "motion_detection_backend": "cv2_frame_differencing",
        "motion_detection_device_provider": "cpu",
    }

    measured_record.ips_processed = True
    measured_record.trace_status = "processed"
    measured_record.rpm_total_ms = 40.0
    measured_record.queue_wait_ms = 5.0
    measured_record.object_detection_total_ms = 10.0
    measured_record.object_detection_call_count = 1
    measured_record.face_detection_total_ms = 0.0
    measured_record.face_detection_call_count = 0
    measured_record.face_recognition_total_ms = 0.0
    measured_record.face_recognition_call_count = 0
    measured_record.persons_count = 2
    measured_record.person_rois_count = 2
    measured_record.object_detection_model_path = "models/object_detection/yolo11m.pt"
    measured_record.object_detection_model_name = "yolo11m.pt"
    measured_record.object_detection_device_provider = "ultralytics:cpu"
    measured_record.object_detection_input_size = "640x640"
    measured_record.object_detection_confidence_threshold = "0.25"
    measured_record.object_detection_nms_threshold = "0.45"
    measured_record.rpm_frame_metrics = {
        "is_warmup": False,
        "face_detection_backend": "onnxruntime",
        "face_detection_device_provider": "CPUExecutionProvider",
        "face_detection_model_path": "models/face_detection/det_500m.onnx",
        "face_detection_input_size": "640x640",
        "face_detection_confidence_threshold": "0.5",
        "face_recognition_backend": "onnxruntime",
        "face_recognition_device_provider": "CPUExecutionProvider",
        "face_recognition_model_path": "models/face_recognition/w600k_mbf.onnx",
        "face_recognition_embedding_model": "w600k_mbf.onnx",
        "motion_detection_backend": "cv2_frame_differencing",
        "motion_detection_device_provider": "cpu",
    }

    summary = _build_runtime_trace_summary([warmup_record, measured_record])

    assert summary["total_frames_submitted"] == 1
    assert summary["warmup_frames_submitted"] == 1
    assert summary["warmup_frames_processed"] == 1
    assert summary["first_measured_frame_rpm_total_ms"] == pytest.approx(40.0)
    assert summary["warmup_section"]["warmup_first_frame_rpm_total_ms"] == pytest.approx(120.0)
    assert summary["average_latency_ms_by_stage"]["face_detection_total_ms"] == pytest.approx(0.0)
    assert summary["average_face_detection_model_ms_over_calls"] == pytest.approx(0.0)
    assert summary["average_face_recognition_model_ms_over_calls"] == pytest.approx(0.0)
    assert summary["backend_diagnostics"]["object_detection"]["backend"] == "ultralytics"
    assert summary["backend_diagnostics"]["face_detection"]["backend"] == "onnxruntime"
    assert summary["backend_diagnostics"]["face_recognition"]["embedding_model"] == "w600k_mbf.onnx"
    assert summary["backend_diagnostics"]["motion_detection"]["backend"] == "cv2_frame_differencing"


def test_runtime_trace_per_frame_payload_reports_counts_calls_and_totals() -> None:
    """Per-frame payload must include algorithm counts, calls, and timing totals."""
    frame = cv2.imread(str(get_asset_path("frame_000008.jpg")), cv2.IMREAD_COLOR)
    assert frame is not None
    record = ReplayFrameRecord(
        frame_index=1,
        frame_id="frame-1",
        timestamp_ms=1000,
        original_frame=frame,
        submitted_monotonic_s=0.0,
        submitted_at_ms=1000,
        ips_enqueued_at_ms=1005,
        is_warmup=False,
        motion_stage_ran=True,
        final_motion_regions_count=2,
        object_detection_call_count=2,
        persons_count=5,
        face_detection_call_count=4,
        face_rois_count=3,
        face_recognition_call_count=3,
        recognized_faces_count=2,
        motion_detection_ms=10.0,
        object_detection_total_ms=30.0,
        face_detection_total_ms=20.0,
        face_recognition_total_ms=15.0,
    )

    payload = _build_runtime_trace_per_frame_payload([record])
    assert payload["frames_total"] == 1
    assert len(payload["frames"]) == 1
    frame_payload = payload["frames"][0]
    assert frame_payload["queue_to_aps"]["queue_to_aps_latency_ms"] == pytest.approx(5.0)
    assert frame_payload["motion_detection"]["moving_zones_detected_count"] == 2
    assert frame_payload["motion_detection"]["algorithm_call_count"] == 1
    assert frame_payload["object_detection"]["detected_rectangles_count"] == 5
    assert frame_payload["object_detection"]["algorithm_call_count"] == 2
    assert frame_payload["face_detection"]["detected_face_zones_count"] == 3
    assert frame_payload["face_detection"]["algorithm_call_count"] == 4
    assert frame_payload["face_recognition"]["recognized_faces_count"] == 2
    assert frame_payload["face_recognition"]["algorithm_call_count"] == 3
    assert frame_payload["totals_for_frame"]["total_events_all_algorithms"] == 12
    assert frame_payload["totals_for_frame"]["total_algorithm_calls_all_algorithms"] == 10
    assert frame_payload["totals_for_frame"]["total_time_all_algorithms_ms"] == pytest.approx(75.0)


@pytest.mark.integration
def test_visual_replay_writes_side_by_side_video(tmp_path: Path) -> None:
    """Write the optional side-by-side output alongside the overlay video."""
    video_path = tmp_path / "single_camera_test.mp4"
    output_dir = tmp_path / "output"
    _write_temp_video(video_path)

    result = run_visual_replay(
        input_video_path=video_path,
        output_dir=output_dir,
        replay_fps=_FPS,
        enable_side_by_side=True,
    )

    assert result.output_overlay_path.exists()
    assert result.output_side_by_side_path is not None
    assert result.output_side_by_side_path.exists()


@pytest.mark.integration
def test_visual_replay_quality_mode_uses_slower_replay_and_relaxed_freshness(
    tmp_path: Path,
) -> None:
    """Run the real replay path in quality mode and write the expected artifacts."""
    video_path = tmp_path / "single_camera_test.mp4"
    output_dir = tmp_path / "quality_output"
    _write_temp_video(video_path)

    result = run_visual_replay(
        input_video_path=video_path,
        output_dir=output_dir,
        quality_mode=True,
    )

    assert result.output_overlay_path.exists()
    assert result.summary_path.exists()
    assert result.metrics_csv_path.exists()
    assert result.frames_read == _FRAME_COUNT
    assert result.frames_submitted == _FRAME_COUNT
    assert result.frames_rendered == _FRAME_COUNT
    assert result.final_service_state in {"STOPPED", "DEGRADED"}


@pytest.mark.integration
def test_visual_replay_warmup_frames_are_reported_separately(tmp_path: Path) -> None:
    """Warmup frames must be excluded from measured totals and marked in runtime trace output."""
    video_path = tmp_path / "single_camera_test.mp4"
    output_dir = tmp_path / "output"
    _write_temp_video(video_path)

    result = run_visual_replay(
        input_video_path=video_path,
        output_dir=output_dir,
        replay_fps=_FPS,
        warmup_frames=1,
    )

    assert result.warmup_frames_submitted == 1
    assert result.frames_submitted == 1

    runtime_trace_path = output_dir / "runtime_trace.csv"
    runtime_trace_summary_path = output_dir / "runtime_trace_summary.json"
    summary_path = output_dir / "summary.json"

    with open(runtime_trace_path, newline="", encoding="utf-8") as handle:
        trace_rows = list(csv.DictReader(handle))
    assert len(trace_rows) == 2
    assert "is_warmup" in trace_rows[0]
    assert sum(1 for row in trace_rows if row.get("is_warmup") == "True") == 1

    with open(runtime_trace_summary_path, encoding="utf-8") as handle:
        trace_summary = json.load(handle)
    assert trace_summary["warmup_frames_submitted"] == 1
    assert trace_summary["total_frames_submitted"] == 1
    assert "warmup_section" in trace_summary
    assert trace_summary["backend_diagnostics"]["object_detection"]["backend"] == "unknown"
    assert trace_summary["backend_diagnostics"]["face_detection"]["backend"] == "onnxruntime"
    assert trace_summary["backend_diagnostics"]["face_recognition"]["backend"] == "onnxruntime"
    assert trace_summary["backend_diagnostics"]["motion_detection"]["backend"] == "cv2_frame_differencing"

    with open(summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["warmup_frames_submitted"] == 1
    assert "backend_diagnostics" in summary
