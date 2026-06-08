"""Single-camera visual replay runner for the Image Processing Service."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("opencv-python is required for visual replay") from exc

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("numpy is required for visual replay") from exc

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _PROJECT_ROOT / "src"
_TESTS_DIR = _PROJECT_ROOT / "tests"
for _path in (_PROJECT_ROOT, _SRC_DIR, _TESTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from image_processing.frame_ingestion_gateway import (  # type: ignore[import-not-found]
    InMemoryFrameIngressTransport,
    IngressFrameMessage,
)
from image_processing.image_processing_service import (  # type: ignore[import-not-found]
    ImageProcessingService,
)
from image_processing_service.ips_yaml_config_helpers import (  # type: ignore[import-not-found]
    build_ips_config_from_yaml,
)
from scripts.run_rpm_on_frames import (  # type: ignore[import-not-found]
    build_runtime_components,
    load_yaml_config,
)

CAMERA_ID = "camera_1"
DEFAULT_INPUT_VIDEO_PATH = (
    _PROJECT_ROOT
    / "tests"
    / "visual_assets"
    / "image_processing_service"
    / "videos"
    / "single_camera_test.mp4"
)
EXISTING_INPUT_VIDEO_PATH = (
    _PROJECT_ROOT
    / "tests"
    / "image_processing_service"
    / "videos"
    / "single_camera_test.mp4"
)
DEFAULT_OUTPUT_DIR = (
    _PROJECT_ROOT
    / "tests"
    / "results"
    / "image_processing_service_visual"
    / "single_camera"
)
QUALITY_OUTPUT_DIR = (
    _PROJECT_ROOT
    / "tests"
    / "results"
    / "image_processing_service_visual"
    / "single_camera_quality_mode"
)
DEFAULT_RPM_CONFIG_PATH = (
    _PROJECT_ROOT / "config" / "image_processing_service" / "local_debug.yaml"
)
DEFAULT_IPS_CONFIG_PATH = (
    _PROJECT_ROOT
    / "config"
    / "image_processing_service"
    / "image_processing_service.yaml"
)
DEFAULT_PERSON_JSON_PATH = _PROJECT_ROOT / "data" / "person_directory.json"
DEFAULT_REPLAY_FPS = 5.0
DEFAULT_WARMUP_FRAMES = 0
QUALITY_REPLAY_FPS = 1.0
DEFAULT_DRAIN_TIMEOUT_S = 30.0
QUALITY_MAX_QUEUE_SIZE_PER_CAMERA = 1000
QUALITY_MAX_QUEUE_BYTES_PER_CAMERA = 1_073_741_824
QUALITY_MAX_TOTAL_QUEUED_BYTES = 2_147_483_648
QUALITY_MAX_FRAME_AGE_MS = 120_000
FOURCC_MP4V = "mp4v"
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
HUD_LINE_HEIGHT = 18
HUD_ORIGIN_X = 8
HUD_ORIGIN_Y = 22
HUD_TEXT_COLOR = (235, 235, 235)
STATUS_TEXT_COLOR = (20, 20, 255)
MOTION_ROI_COLOR = (0, 255, 255)  # yellow — motion region bounding boxes
PERSON_COLOR = (0, 255, 0)
FACE_COLOR = (255, 255, 0)
PANEL_LABEL_COLOR = (255, 255, 255)
BORDER_THICKNESS = 2
DEMO_OVERLAY_VIDEO_NAME = "demo_annotated.mp4"
DEMO_CAMERA_LABEL = "CAM_01"
DEMO_SUMMARY_SCREEN_SECONDS = 0.8
DEMO_FRAME_FONT_SCALE = 0.78
DEMO_FRAME_FONT_THICKNESS = 2
DEMO_HUD_FONT_SCALE = 0.72
DEMO_HUD_TITLE_FONT_SCALE = 0.88
DEMO_HUD_THICKNESS = 2
SUMMARY_TITLE_FONT_SCALE = 0.94
SUMMARY_SECTION_FONT_SCALE = 0.72
SUMMARY_DETAIL_FONT_SCALE = 0.62
SUMMARY_CARD_ALPHA = 0.84
SUMMARY_BACKGROUND_COLOR = (12, 12, 12)
SUMMARY_CARD_COLOR = (28, 28, 28)
SUMMARY_TEXT_COLOR = (240, 240, 240)
SUMMARY_ACCENT_COLOR = (180, 180, 180)
SUMMARY_SUCCESS_COLOR = (0, 220, 0)
MOTION_BBOX_THICKNESS = 3
PERSON_BBOX_THICKNESS = 4
FACE_BBOX_THICKNESS = 3
RECOGNITION_LABEL_THICKNESS = 2
SAMPLE_FRAME_STEM = "frame"
OVERLAY_VIDEO_NAME = "output_overlay.mp4"
SIDE_BY_SIDE_VIDEO_NAME = "output_side_by_side.mp4"
SUMMARY_NAME = "summary.json"
METRICS_NAME = "metrics.csv"
RUNTIME_TRACE_NAME = "runtime_trace.csv"
RUNTIME_TRACE_SUMMARY_NAME = "runtime_trace_summary.json"
RUNTIME_TRACE_PER_FRAME_NAME = "runtime_trace_per_frame.json"
SAMPLE_FRAMES_DIR_NAME = "sample_frames"


class VisualReplayError(RuntimeError):
    """Raised when the visual replay runner cannot complete."""


@dataclass(slots=True)
class ReplayFrameRecord:
    """In-memory record for one replayed frame and its observed runtime state."""

    frame_index: int
    frame_id: str
    timestamp_ms: int
    original_frame: np.ndarray
    submitted_monotonic_s: float
    queue_depth: int = 0
    service_state: str = "INITIALIZING"
    worker_lag_ms: int = 0
    output: dict[str, Any] | None = None
    processing_latency_ms: float | None = None
    status: str = "submitted"
    stale_status: str = "false"
    dropped_status: str = "false"
    persons_count: int = 0
    faces_count: int = 0
    stale_frames_dropped_total: int = 0
    dropped_frames_total: int = 0
    submitted_to_gateway: bool = False
    gateway_published: bool = False
    ips_enqueued: bool = False
    ips_dequeued: bool = False
    ips_processed: bool = False
    submitted_at_ms: int = 0
    gateway_published_at_ms: int = 0
    ips_enqueued_at_ms: int = 0
    ips_dequeued_at_ms: int = 0
    rpm_started_at_ms: int = 0
    rpm_completed_at_ms: int = 0
    rendered_at_ms: int = 0
    trace_status: str = "skipped"
    stop_reason: str = "dropped_before_processing"
    stop_reason_from_orchestrator: str = ""
    stage_path: str = "NOT_PROCESSED"
    motion_stage_ran: bool = False
    motion_detected: bool = False
    motion_regions_count: int = 0
    object_detection_stage_ran: bool = False
    person_rois_count: int = 0
    face_detection_stage_ran: bool = False
    face_rois_count: int = 0
    face_recognition_stage_ran: bool = False
    recognized_faces_count: int = 0
    gateway_ingest_ms: float | None = None
    queue_wait_ms: float | None = None
    ips_worker_wait_ms: float | None = None
    worker_pre_rpm_delay_ms: float | None = None
    rpm_total_ms: float | None = None
    motion_detection_ms: float | None = None
    ftl_ingest_ms: float | None = None
    ftl_get_frame_total_ms: float | None = None
    ftl_get_frame_call_count: int = 0
    object_detection_total_ms: float | None = None
    object_detection_call_count: int = 0
    object_detection_avg_call_ms: float | None = None
    object_detection_max_call_ms: float | None = None
    object_detection_roi_width: float | None = None
    object_detection_roi_height: float | None = None
    object_detection_roi_area: float | None = None
    object_detection_largest_roi_area: int = 0
    object_detection_input_size: str = "unavailable"
    object_detection_model_path: str = "unavailable"
    object_detection_model_name: str = "unavailable"
    object_detection_device_provider: str = "unavailable"
    object_detection_confidence_threshold: str = "unavailable"
    object_detection_nms_threshold: str = "unavailable"
    persons_returned_per_call: str = ""
    face_detection_total_ms: float | None = None
    face_detection_call_count: int = 0
    face_detection_avg_call_ms: float | None = None
    face_detection_max_call_ms: float | None = None
    face_detection_roi_width: float | None = None
    face_detection_roi_height: float | None = None
    face_detection_roi_area: float | None = None
    face_detection_largest_roi_area: int = 0
    face_recognition_total_ms: float | None = None
    face_recognition_call_count: int = 0
    person_directory_lookup_total_ms: float | None = None
    person_directory_lookup_count: int = 0
    output_build_ms: float | None = None
    motion_regions_dropped_by_cap: int = 0
    raw_motion_regions_count: int = 0
    filtered_motion_regions_count: int = 0
    merged_motion_regions_count: int = 0
    final_motion_regions_count: int = 0
    motion_regions_dropped_by_filter: int = 0
    motion_regions_dropped_by_merge: int = 0
    motion_roi_areas: str = ""  # JSON-serialized list[int]
    motion_roi_largest_area: int = 0
    motion_roi_smallest_area: int = 0
    motion_roi_total_area: int = 0
    person_rois_dropped_by_cap: int = 0
    face_rois_dropped_by_cap: int = 0
    gate_violations: str = ""
    slowest_stage: str = ""
    motion_roi_bboxes: list[dict[str, Any]] = field(default_factory=list)
    demo_bypass_motion_gate: bool = False
    total_end_to_end_ms: float | None = None
    pipeline_error: bool = False
    is_warmup: bool = False
    rpm_frame_metrics: dict[str, Any] | None = None


@dataclass(slots=True)
class VisualReplayRunResult:
    """Structured result returned by the visual replay runner."""

    output_overlay_path: Path
    summary_path: Path
    metrics_csv_path: Path
    output_side_by_side_path: Path | None
    sample_frames_dir: Path | None
    frames_read: int
    frames_submitted: int
    frames_processed: int
    warmup_frames_submitted: int
    warmup_frames_processed: int
    frames_rendered: int
    frames_dropped: int
    stale_frames_dropped: int
    average_processing_latency_ms: float
    max_processing_latency_ms: float
    total_person_detections: int
    total_face_detections: int
    total_motion_regions_detected: int
    total_recognized_faces: int
    average_fps: float
    gateway_frames_in_total: int
    gateway_frames_published_total: int
    ips_processed_total: int
    ftl_calls_last_frame: int
    final_service_state: str
    final_active_workers: int


class RecordingResultHandler:
    """Capture worker outputs so rendering can happen after shutdown."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[str, tuple[dict[str, Any], float, dict[str, Any]]] = {}
        self._rpm: Any | None = None

    def attach_rpm(self, rpm: Any) -> None:
        """Attach RPM instance used to snapshot per-frame orchestrator metrics."""
        with self._lock:
            self._rpm = rpm

    def handle(
        self,
        camera_id: str,
        frame_packet: Any,
        output: dict[str, Any],
    ) -> None:
        """Record one worker result keyed by frame identifier."""
        del camera_id
        frame_metrics = self._snapshot_frame_metrics()
        with self._lock:
            self._records[frame_packet.frame_id] = (output, time.monotonic(), frame_metrics)

    def get_record(self, frame_id: str) -> tuple[dict[str, Any], float, dict[str, Any]] | None:
        """Return the recorded output for one frame when available."""
        with self._lock:
            return self._records.get(frame_id)

    def processed_count(self) -> int:
        """Return the number of processed frames captured by the handler."""
        with self._lock:
            return len(self._records)

    def _snapshot_frame_metrics(self) -> dict[str, Any]:
        with self._lock:
            rpm = self._rpm
        if rpm is None:
            return {}
        getter = getattr(rpm, "get_last_frame_metrics", None)
        if not callable(getter):
            return {}
        try:
            value = getter()
        except Exception:
            return {}
        return value if isinstance(value, dict) else {}


class RuntimeTraceCollector:
    """Collect non-invasive per-frame stage diagnostics for replay runs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread_local = threading.local()
        self._by_frame: dict[str, dict[str, Any]] = {}
        self._rpm_ref: Any = None
        self._motion_module: Any = None
        # per-frame timestamps for queue wait calculation
        self._enqueued_at_ms: dict[str, int] = {}
        self._rpm_started_at_ms: dict[str, int] = {}

    def install(self, rpm: Any) -> None:
        """Wrap orchestrator stage calls to collect execution trace data."""
        self._rpm_ref = rpm
        orchestrator = getattr(rpm, "_orchestrator", None)
        if orchestrator is None:
            return
        self._wrap_execute(orchestrator)

    def install_process_frame_hook(self, rpm: Any) -> None:
        """Wrap rpm.process_frame to capture per-frame RPM start timestamp."""
        original = getattr(rpm, "process_frame", None)
        if original is None or not callable(original):
            return

        collector = self

        def _hooked(frame_packet: Any) -> Any:
            frame_id = str(frame_packet.frame_id)
            started_ms = int(time.monotonic() * 1000)
            with collector._lock:
                collector._rpm_started_at_ms[frame_id] = started_ms
                trace = collector._by_frame.setdefault(frame_id, _new_trace_row())
                trace["rpm_started_at_ms"] = started_ms
                trace["ips_dequeued"] = True
                trace["ips_dequeued_at_ms"] = started_ms
            return original(frame_packet)

        rpm.process_frame = _hooked

    def install_sink_hook(self, sink: Any) -> None:
        """Wrap sink.enqueue to record per-frame IPS enqueue timestamp."""
        original = getattr(sink, "enqueue", None)
        if original is None or not callable(original):
            return

        collector = self

        def _hooked(frame_packet: Any) -> Any:
            frame_id = str(frame_packet.frame_id)
            enqueued_ms = int(time.monotonic() * 1000)
            with collector._lock:
                collector._enqueued_at_ms[frame_id] = enqueued_ms
                trace = collector._by_frame.setdefault(frame_id, _new_trace_row())
                trace["gateway_published"] = True
                trace["ips_enqueued"] = True
                trace["gateway_published_at_ms"] = enqueued_ms
                trace["ips_enqueued_at_ms"] = enqueued_ms
            return original(frame_packet)

        sink.enqueue = _hooked

    def install_motion_capture(self, motion_module: Any) -> None:
        """Register the logging motion module to capture per-frame motion bboxes."""
        self._motion_module = motion_module

    def get_queue_wait_ms(self, frame_id: str) -> float | None:
        """Return queue_wait_ms for a frame if both timestamps are available."""
        with self._lock:
            enqueued = self._enqueued_at_ms.get(frame_id)
            started = self._rpm_started_at_ms.get(frame_id)
        if enqueued is not None and started is not None and started >= enqueued:
            return float(started - enqueued)
        return None

    def mark_submitted(self, frame_id: str, frame_index: int, camera_id: str) -> None:
        """Ensure a trace row exists for every submitted frame."""
        now_ms = int(time.monotonic() * 1000)
        with self._lock:
            trace = self._by_frame.setdefault(frame_id, _new_trace_row())
            trace["frame_id"] = frame_id
            trace["frame_index"] = frame_index
            trace["camera_id"] = camera_id
            trace["submitted_to_gateway"] = True
            trace["submitted_at_ms"] = now_ms

    def mark_warmup(self, frame_id: str, is_warmup: bool) -> None:
        """Mark one frame trace row as belonging to the warmup phase."""
        with self._lock:
            trace = self._by_frame.setdefault(frame_id, _new_trace_row())
            trace["is_warmup"] = bool(is_warmup)

    def get_trace_for_frame(self, frame_id: str) -> dict[str, Any]:
        """Return collected stage trace for one frame id."""
        with self._lock:
            return dict(self._by_frame.get(frame_id, _new_trace_row()))

    def _wrap_execute(self, orchestrator: Any) -> None:
        execute = getattr(orchestrator, "execute", None)
        if execute is None or not callable(execute):
            return

        def wrapped(frame_packet: Any) -> Any:
            frame_id = str(frame_packet.frame_id)
            dequeued_ms = int(time.monotonic() * 1000)
            self._set_current_frame(frame_id)
            self._update(
                frame_id,
                {
                    "rpm_frame_ran": True,
                    "ips_dequeued": True,
                    "ips_dequeued_at_ms": dequeued_ms,
                },
            )
            start = time.perf_counter()
            try:
                result = execute(frame_packet)
            except Exception:
                self._update(frame_id, {"pipeline_error": True})
                raise
            finally:
                elapsed = (time.perf_counter() - start) * 1000.0
                completed_ms = int(time.monotonic() * 1000)
                self._accumulate(frame_id, "rpm_total_ms", elapsed)
                self._update(frame_id, {"rpm_completed_at_ms": completed_ms})
                self._clear_current_frame()
            metrics_getter = getattr(orchestrator, "get_last_frame_metrics", None)
            metrics = metrics_getter() if callable(metrics_getter) else {}
            metrics = metrics if isinstance(metrics, dict) else {}
            # --- real per-stage timing (from orchestrator frame_metrics) ---
            motion_stage_ran = bool(metrics.get("motion_stage_ran", False))
            motion_detected = bool(metrics.get("motion_detected", False))
            motion_regions = int(metrics.get("motion_bboxes_after_merge_count", 0))
            od_call_count = int(metrics.get("object_detection_call_count", 0))
            fd_call_count = int(metrics.get("face_detection_call_count", 0))
            fr_call_count = int(metrics.get("face_recognition_call_count", 0))
            person_rois_raw = int(metrics.get("person_rois_raw_count", 0))
            face_rois_raw = int(metrics.get("face_rois_raw_count", 0))
            recog_faces = int(metrics.get("recognized_faces_count", 0))
            gate_violations = list(metrics.get("gate_violations", []))
            orch_stop_reason = str(metrics.get("stop_reason", ""))
            slowest = _slowest_stage_from_metrics(metrics)
            queue_wait_ms = self.get_queue_wait_ms(frame_id)
            # collect motion region bboxes from the logging motion wrapper
            captured_motion_bboxes: list[dict[str, Any]] = []
            if self._motion_module is not None:
                flush = getattr(self._motion_module, "flush_calls", None)
                if callable(flush):
                    for call in flush():
                        for bbox in call.get("output", {}).get("bboxes", []):
                            if isinstance(bbox, dict):
                                captured_motion_bboxes.append(bbox)
            # also try to read output_build_ms from rpm directly
            build_ms_getter = getattr(self._rpm_ref, "get_last_output_build_ms", None)
            output_build_ms = 0.0
            if callable(build_ms_getter):
                try:
                    output_build_ms = float(build_ms_getter())
                except Exception:
                    output_build_ms = 0.0
            self._update(
                frame_id,
                {
                    # stage run flags (now sourced from real metrics)
                    "motion_stage_ran": motion_stage_ran,
                    "motion_detected": motion_detected,
                    "motion_regions_count": motion_regions,
                    "object_detection_stage_ran": od_call_count > 0,
                    "person_rois_count": float(person_rois_raw),
                    "face_detection_stage_ran": fd_call_count > 0,
                    "face_recognition_stage_ran": fr_call_count > 0,
                    "face_rois_count": float(face_rois_raw),
                    "recognized_faces_count": float(recog_faces),
                    # real per-stage timings
                    "ftl_ingest_ms": float(metrics.get("ftl_ingest_ms", 0.0)),
                    "ftl_get_frame_total_ms": float(metrics.get("ftl_get_frame_total_ms", 0.0)),
                    "ftl_get_frame_call_count": int(metrics.get("ftl_get_frame_call_count", 0)),
                    "motion_detection_ms": float(metrics.get("motion_detection_ms", 0.0)),
                    "object_detection_total_ms": float(metrics.get("object_detection_total_ms", 0.0)),
                    "object_detection_call_count": od_call_count,
                    "object_detection_avg_call_ms": float(
                        metrics.get("object_detection_avg_call_ms", 0.0)
                    ),
                    "object_detection_max_call_ms": float(
                        metrics.get("object_detection_max_call_ms", 0.0)
                    ),
                    "object_detection_roi_width": float(
                        metrics.get("object_detection_roi_width", 0.0)
                    ),
                    "object_detection_roi_height": float(
                        metrics.get("object_detection_roi_height", 0.0)
                    ),
                    "object_detection_roi_area": float(
                        metrics.get("object_detection_roi_area", 0.0)
                    ),
                    "object_detection_largest_roi_area": int(
                        metrics.get("object_detection_largest_roi_area", 0)
                    ),
                    "object_detection_input_size": str(
                        metrics.get("object_detection_input_size", "unavailable")
                    ),
                    "object_detection_model_path": str(
                        metrics.get("object_detection_model_path", "unavailable")
                    ),
                    "object_detection_model_name": str(
                        metrics.get("object_detection_model_name", "unavailable")
                    ),
                    "object_detection_device_provider": str(
                        metrics.get("object_detection_device_provider", "unavailable")
                    ),
                    "object_detection_confidence_threshold": str(
                        metrics.get("object_detection_confidence_threshold", "unavailable")
                    ),
                    "object_detection_nms_threshold": str(
                        metrics.get("object_detection_nms_threshold", "unavailable")
                    ),
                    "persons_returned_per_call": _serialize_int_list(
                        metrics.get("persons_returned_per_call", [])
                    ),
                    "face_detection_total_ms": float(metrics.get("face_detection_total_ms", 0.0)),
                    "face_detection_call_count": fd_call_count,
                    "face_detection_avg_call_ms": float(
                        metrics.get("face_detection_avg_call_ms", 0.0)
                    ),
                    "face_detection_max_call_ms": float(
                        metrics.get("face_detection_max_call_ms", 0.0)
                    ),
                    "face_detection_roi_width": float(
                        metrics.get("face_detection_roi_width", 0.0)
                    ),
                    "face_detection_roi_height": float(
                        metrics.get("face_detection_roi_height", 0.0)
                    ),
                    "face_detection_roi_area": float(
                        metrics.get("face_detection_roi_area", 0.0)
                    ),
                    "face_detection_largest_roi_area": int(
                        metrics.get("face_detection_largest_roi_area", 0)
                    ),
                    "face_recognition_total_ms": float(metrics.get("face_recognition_total_ms", 0.0)),
                    "face_recognition_call_count": fr_call_count,
                    "person_directory_lookup_total_ms": float(
                        metrics.get("person_directory_lookup_total_ms", 0.0)
                    ),
                    "person_directory_lookup_count": int(
                        metrics.get("person_directory_lookup_count", 0)
                    ),
                    "output_build_ms": output_build_ms,
                    "queue_wait_ms": queue_wait_ms if queue_wait_ms is not None else 0.0,
                    # fan-out drop counts
                    "motion_regions_dropped_by_cap": int(
                        metrics.get("motion_regions_dropped_by_cap", 0)
                    ),
                    "raw_motion_regions_count": int(
                        metrics.get("motion_bboxes_raw_count", 0)
                    ),
                    "filtered_motion_regions_count": int(
                        metrics.get("motion_bboxes_after_filter_count", 0)
                    ),
                    "merged_motion_regions_count": int(
                        metrics.get("motion_bboxes_after_merge_count", 0)
                    ),
                    "final_motion_regions_count": int(
                        metrics.get("final_motion_regions_count", 0)
                    ),
                    "motion_regions_dropped_by_filter": int(
                        metrics.get("motion_regions_dropped_by_filter", 0)
                    ),
                    "motion_regions_dropped_by_merge": int(
                        metrics.get("motion_regions_dropped_by_merge", 0)
                    ),
                    "motion_roi_areas": _serialize_int_list(
                        metrics.get("motion_roi_areas", [])
                    ),
                    "motion_roi_largest_area": int(
                        metrics.get("motion_roi_largest_area", 0)
                    ),
                    "motion_roi_smallest_area": int(
                        metrics.get("motion_roi_smallest_area", 0)
                    ),
                    "motion_roi_total_area": int(
                        metrics.get("motion_roi_total_area", 0)
                    ),
                    "person_rois_dropped_by_cap": int(
                        metrics.get("person_rois_dropped_by_cap", 0)
                    ),
                    "face_rois_dropped_by_cap": int(
                        metrics.get("face_rois_dropped_by_cap", 0)
                    ),
                    # gate violations & stop reason from orchestrator
                    "gate_violations": "|".join(gate_violations) if gate_violations else "",
                    "stop_reason_from_orchestrator": orch_stop_reason,
                    "slowest_stage": slowest,
                    "motion_roi_bboxes": captured_motion_bboxes,
                    "demo_bypass_motion_gate": bool(
                        metrics.get("demo_bypass_motion_gate", False)
                    ),
                },
            )
            return result

        setattr(orchestrator, "execute", wrapped)

    def _set_current_frame(self, frame_id: str) -> None:
        self._thread_local.current_frame_id = frame_id

    def _clear_current_frame(self) -> None:
        self._thread_local.current_frame_id = None

    def _current_frame_id(self, fallback: str) -> str:
        frame_id = getattr(self._thread_local, "current_frame_id", None)
        return fallback if not frame_id else str(frame_id)

    def _update(self, frame_id: str, updates: dict[str, Any]) -> None:
        with self._lock:
            trace = self._by_frame.setdefault(frame_id, _new_trace_row())
            trace.update(updates)

    def _accumulate(self, frame_id: str, key: str, delta: float) -> None:
        with self._lock:
            trace = self._by_frame.setdefault(frame_id, _new_trace_row())
            trace[key] = float(trace.get(key, 0.0)) + float(delta)


_STAGE_TIMING_KEYS: list[str] = [
    "ftl_ingest_ms",
    "ftl_get_frame_total_ms",
    "motion_detection_ms",
    "object_detection_total_ms",
    "face_detection_total_ms",
    "face_recognition_total_ms",
    "person_directory_lookup_total_ms",
    "output_build_ms",
]


def _slowest_stage_from_metrics(metrics: dict[str, Any]) -> str:
    """Return the name of the slowest stage based on metrics timing fields."""
    best_key = ""
    best_val = -1.0
    for key in _STAGE_TIMING_KEYS:
        val = float(metrics.get(key, 0.0))
        if val > best_val:
            best_val = val
            best_key = key
    return best_key if best_val > 0.0 else ""


def _new_trace_row() -> dict[str, Any]:
    """Return a new mutable runtime trace row with defaults."""
    return {
        "frame_id": "",
        "frame_index": -1,
        "camera_id": CAMERA_ID,
        "is_warmup": False,
        "submitted_to_gateway": False,
        "submitted_at_ms": 0,
        "gateway_published": False,
        "gateway_published_at_ms": 0,
        "ips_enqueued": False,
        "ips_enqueued_at_ms": 0,
        "ips_dequeued": False,
        "ips_dequeued_at_ms": 0,
        "ips_processed": False,
        "rpm_started_at_ms": 0,
        "rpm_completed_at_ms": 0,
        "rendered_at_ms": 0,
        "rpm_frame_ran": False,
        "motion_stage_ran": False,
        "motion_detected": False,
        "motion_regions_count": 0,
        "object_detection_stage_ran": False,
        "person_rois_count": 0.0,
        "face_detection_stage_ran": False,
        "face_rois_count": 0.0,
        "face_recognition_stage_ran": False,
        "recognized_faces_count": 0.0,
        "gateway_ingest_ms": 0.0,
        "queue_wait_ms": 0.0,
        "ips_worker_wait_ms": 0.0,
        "worker_pre_rpm_delay_ms": 0.0,
        "rpm_total_ms": 0.0,
        "motion_detection_ms": 0.0,
        "ftl_ingest_ms": 0.0,
        "ftl_get_frame_total_ms": 0.0,
        "ftl_get_frame_call_count": 0,
        "object_detection_total_ms": 0.0,
        "object_detection_call_count": 0,
        "object_detection_avg_call_ms": 0.0,
        "object_detection_max_call_ms": 0.0,
        "object_detection_roi_width": 0.0,
        "object_detection_roi_height": 0.0,
        "object_detection_roi_area": 0.0,
        "object_detection_largest_roi_area": 0,
        "object_detection_input_size": "unavailable",
        "object_detection_model_path": "unavailable",
        "object_detection_model_name": "unavailable",
        "object_detection_device_provider": "unavailable",
        "object_detection_confidence_threshold": "unavailable",
        "object_detection_nms_threshold": "unavailable",
        "persons_returned_per_call": "",
        "face_detection_total_ms": 0.0,
        "face_detection_call_count": 0,
        "face_detection_avg_call_ms": 0.0,
        "face_detection_max_call_ms": 0.0,
        "face_detection_roi_width": 0.0,
        "face_detection_roi_height": 0.0,
        "face_detection_roi_area": 0.0,
        "face_detection_largest_roi_area": 0,
        "face_recognition_total_ms": 0.0,
        "face_recognition_call_count": 0,
        "person_directory_lookup_total_ms": 0.0,
        "person_directory_lookup_count": 0,
        "output_build_ms": 0.0,
        "motion_regions_dropped_by_cap": 0,
        "raw_motion_regions_count": 0,
        "filtered_motion_regions_count": 0,
        "merged_motion_regions_count": 0,
        "final_motion_regions_count": 0,
        "motion_regions_dropped_by_filter": 0,
        "motion_regions_dropped_by_merge": 0,
        "motion_roi_areas": "",
        "motion_roi_largest_area": 0,
        "motion_roi_smallest_area": 0,
        "motion_roi_total_area": 0,
        "person_rois_dropped_by_cap": 0,
        "face_rois_dropped_by_cap": 0,
        "gate_violations": "",
        "slowest_stage": "",
        "stop_reason_from_orchestrator": "",
        "pipeline_error": False,
        "motion_roi_bboxes": [],
        "demo_bypass_motion_gate": False,
    }


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the replay runner."""
    parser = argparse.ArgumentParser(
        description="Run the single-camera IPS visual replay.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-video",
        type=Path,
        default=None,
        help=(
            "Input mp4 replay file. When omitted, the runner first checks "
            "the documented default path, then the existing tests path."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for replay artifacts.",
    )
    parser.add_argument(
        "--replay-fps",
        type=float,
        default=DEFAULT_REPLAY_FPS,
        help="Submission FPS used to pace ingress replay.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=DEFAULT_WARMUP_FRAMES,
        help="Number of warmup frames to run before measured replay.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to replay.",
    )
    parser.add_argument(
        "--side-by-side",
        action="store_true",
        help="Also write left-original/right-overlay video output.",
    )
    parser.add_argument(
        "--sample-frame-interval",
        type=int,
        default=0,
        help="Write sample overlay PNGs every N rendered frames when > 0.",
    )
    parser.add_argument(
        "--quality-mode",
        action="store_true",
        help=(
            "Run the replay with slower ingress and relaxed freshness limits "
            "to prioritize processing completeness."
        ),
    )
    parser.add_argument(
        "--debug-trace-hud",
        action="store_true",
        help="Render stage-path runtime trace lines in the overlay HUD.",
    )
    parser.add_argument(
        "--rpm-config-path",
        type=Path,
        default=DEFAULT_RPM_CONFIG_PATH,
        help="RPM YAML config path used to build runtime components.",
    )
    parser.add_argument(
        "--ips-config-path",
        type=Path,
        default=DEFAULT_IPS_CONFIG_PATH,
        help="IPS YAML config path used to configure the service.",
    )
    parser.add_argument(
        "--demo-render",
        action="store_true",
        help="Write the polished demo video with the summary screen.",
    )
    parser.add_argument(
        "--demo-summary-seconds",
        type=float,
        default=DEMO_SUMMARY_SCREEN_SECONDS,
        help="Duration of the appended summary screen when --demo-render is enabled.",
    )
    parser.add_argument(
        "--demo-bypass-motion-gate",
        action="store_true",
        help=(
            "Demo-only fallback: if motion returns zero regions, run object detection on a "
            "full-frame ROI instead of stopping at no_motion. Disabled by default."
        ),
    )
    return parser.parse_args()


def _build_missing_video_message() -> str:
    """Return the actionable error message for a missing replay video."""
    return (
        "Missing input video for IPS visual replay. "
        "Expected one of these paths: "
        f"{DEFAULT_INPUT_VIDEO_PATH} or {EXISTING_INPUT_VIDEO_PATH}. "
        f"Place the replay video at {DEFAULT_INPUT_VIDEO_PATH} "
        "or pass --input-video with a valid mp4 path."
    )


def _require_input_video(video_path: Path) -> None:
    """Validate that the requested replay video exists."""
    if not video_path.exists():
        raise VisualReplayError(f"Missing input video for IPS visual replay: {video_path}")
    if not video_path.is_file():
        raise VisualReplayError(
            f"Replay input path is not a file: {video_path}. "
            "Provide a valid mp4 file path."
        )


def _resolve_input_video_path(input_video_path: Path | None) -> Path:
    """Resolve input video path using default->existing fallback behavior."""
    if input_video_path is not None:
        _require_input_video(input_video_path)
        return input_video_path
    candidate_paths = [DEFAULT_INPUT_VIDEO_PATH, EXISTING_INPUT_VIDEO_PATH]
    for candidate_path in candidate_paths:
        if candidate_path.exists() and candidate_path.is_file():
            return candidate_path
    raise VisualReplayError(_build_missing_video_message())


def _clean_output_directory(output_dir: Path) -> None:
    """Remove stale artifacts and recreate the output directory."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _open_video_capture(video_path: Path) -> cv2.VideoCapture:
    """Open a video capture or raise a replay error."""
    capture = cv2.VideoCapture(str(video_path))
    if capture.isOpened():
        return capture
    raise VisualReplayError(f"Unable to open replay video: {video_path}")


def _resolve_replay_fps(capture: cv2.VideoCapture, replay_fps: float | None) -> float:
    """Resolve the pacing FPS from the caller override or video metadata."""
    if replay_fps is not None and replay_fps > 0:
        return replay_fps
    detected_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if detected_fps > 0:
        return detected_fps
    return DEFAULT_REPLAY_FPS


def _read_video_frames(
    video_path: Path,
    replay_fps: float | None,
    max_frames: int | None,
) -> tuple[list[np.ndarray], float]:
    """Decode replay frames and return them with the pacing FPS."""
    capture = _open_video_capture(video_path)
    frames: list[np.ndarray] = []
    effective_fps = _resolve_replay_fps(capture, replay_fps)
    try:
        while True:
            if max_frames is not None and len(frames) >= max_frames:
                break
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise VisualReplayError(f"Replay video contains no decodable frames: {video_path}")
    return frames, effective_fps


def _resolve_repo_path(config_path: str | Path) -> Path:
    """Resolve an absolute or repository-relative config path."""
    candidate = Path(config_path)
    if candidate.is_absolute():
        return candidate
    return _PROJECT_ROOT / candidate


def _resolve_object_detection_is_real(rpm_config: dict[str, Any]) -> bool:
    """Resolve object detection implementation mode from the configured YAML."""
    models_config = rpm_config.get("models")
    if not isinstance(models_config, dict):
        raise VisualReplayError("RPM config is missing models section")
    od_config_raw = models_config.get("object_detection_config")
    if not isinstance(od_config_raw, str) or not od_config_raw.strip():
        raise VisualReplayError("RPM config is missing models.object_detection_config")
    od_config_path = _resolve_repo_path(od_config_raw.strip())
    od_yaml = load_yaml_config(od_config_path)
    implementation_type = str(od_yaml.get("implementation_type", "stub")).strip().lower()
    is_real = implementation_type == "real"
    print(f"[OD Resolution] Config path: {od_config_path}")
    print(f"[OD Resolution] implementation_type: {implementation_type}")
    print(f"[OD Resolution] mode: {'REAL' if is_real else 'STUB'}")
    return is_real


def _load_runtime_components(
    frame: np.ndarray,
    rpm_config_path: Path,
    demo_bypass_motion_gate: bool,
) -> Any:
    """Build runtime components using frame size and YAML-driven OD mode."""
    frame_height, frame_width = frame.shape[:2]
    rpm_config = load_yaml_config(rpm_config_path)
    real_object_detection = _resolve_object_detection_is_real(rpm_config)
    return build_runtime_components(
        rpm_config=rpm_config,
        person_json=DEFAULT_PERSON_JSON_PATH,
        frame_width=frame_width,
        frame_height=frame_height,
        real_motion_detection=True,
        real_object_detection=real_object_detection,
        real_face_detection=True,
        real_face_recognition=True,
        demo_bypass_motion_gate=demo_bypass_motion_gate,
    )


def _build_service_for_frames(
    frames: list[np.ndarray],
    result_handler: RecordingResultHandler,
    quality_mode: bool,
    rpm_config_path: Path,
    ips_config_path: Path,
    demo_bypass_motion_gate: bool,
) -> tuple[ImageProcessingService, InMemoryFrameIngressTransport, Any]:
    """Create and configure the replay service using actual frame size."""
    components = _load_runtime_components(frames[0], rpm_config_path, demo_bypass_motion_gate)
    transport = InMemoryFrameIngressTransport()
    service = ImageProcessingService(
        rpm=components.rpm,
        transport=transport,
        result_handler=result_handler,
    )
    config_overrides: dict[str, Any] = {}
    if quality_mode:
        config_overrides = {
            "max_queue_size_per_camera": QUALITY_MAX_QUEUE_SIZE_PER_CAMERA,
            "max_queue_bytes_per_camera": QUALITY_MAX_QUEUE_BYTES_PER_CAMERA,
            "max_total_queued_bytes": QUALITY_MAX_TOTAL_QUEUED_BYTES,
            "max_frame_age_ms": QUALITY_MAX_FRAME_AGE_MS,
            "overflow_policy": "DROP_NEWEST",
            "stale_frame_drop_enabled": False,
        }
    service.configure(
        build_ips_config_from_yaml(
            camera_ids=[CAMERA_ID],
            config_path=ips_config_path,
            **config_overrides,
        )
    )
    return service, transport, components


def _build_frame_id(frame_index: int, prefix: str = "frame") -> str:
    """Build the deterministic frame identifier for one replayed frame."""
    return f"camera_1_{prefix}_{frame_index:06d}"


def _build_timestamp_ms(base_timestamp_ms: int, frame_index: int, fps: float) -> int:
    """Build the replay timestamp in milliseconds for one frame index."""
    return base_timestamp_ms + int(round((frame_index * 1000.0) / fps))


def _encode_frame_as_message(
    bgr_frame: np.ndarray,
    frame_id: str,
    timestamp_ms: int,
) -> IngressFrameMessage:
    """Encode one BGR frame as a JPEG ingress message."""
    success, encoded = cv2.imencode(".jpg", bgr_frame)
    if not success:
        raise VisualReplayError(f"Unable to JPEG-encode replay frame: {frame_id}")
    height, width = bgr_frame.shape[:2]
    return IngressFrameMessage(
        frame_id=frame_id,
        camera_id=CAMERA_ID,
        timestamp_ms=timestamp_ms,
        width=width,
        height=height,
        source_format="JPEG",
        payload_bytes=encoded.tobytes(),
    )


def _capture_runtime_snapshot(service: ImageProcessingService) -> tuple[int, str, int]:
    """Capture the queue depth, service state, and worker lag for the camera."""
    health = service.health()
    queue_depth = int(health.queue_depth_per_camera.get(CAMERA_ID, 0))
    last_started = int(health.last_frame_started_at_ms.get(CAMERA_ID, 0))
    last_completed = int(health.last_frame_completed_at_ms.get(CAMERA_ID, 0))
    worker_lag_ms = max(0, last_started - last_completed)
    return queue_depth, str(health.state), worker_lag_ms


def _pace_replay(
    start_monotonic_s: float,
    frame_index: int,
    fps: float,
) -> None:
    """Sleep until the target wall-clock submit time for one frame."""
    target_elapsed_s = frame_index / fps
    wait_s = target_elapsed_s - (time.monotonic() - start_monotonic_s)
    if wait_s > 0:
        time.sleep(wait_s)


def _submit_frames(
    frames: list[np.ndarray],
    fps: float,
    service: ImageProcessingService,
    transport: InMemoryFrameIngressTransport,
    trace_collector: RuntimeTraceCollector,
    frame_index_offset: int = 0,
    frame_id_prefix: str = "frame",
    is_warmup: bool = False,
    base_timestamp_ms: int | None = None,
) -> list[ReplayFrameRecord]:
    """Replay decoded frames into the real gateway path at the requested FPS."""
    records: list[ReplayFrameRecord] = []
    resolved_base_timestamp_ms = (
        int(time.time() * 1000) if base_timestamp_ms is None else int(base_timestamp_ms)
    )
    start_monotonic_s = time.monotonic()
    for local_index, frame in enumerate(frames):
        frame_index = frame_index_offset + local_index
        frame_id = _build_frame_id(frame_index, frame_id_prefix)
        timestamp_ms = _build_timestamp_ms(resolved_base_timestamp_ms, frame_index, fps)
        _pace_replay(start_monotonic_s, local_index, fps)
        submitted_monotonic_s = time.monotonic()
        record = ReplayFrameRecord(
            frame_index=frame_index,
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            original_frame=frame.copy(),
            submitted_monotonic_s=submitted_monotonic_s,
            is_warmup=is_warmup,
        )
        record.submitted_to_gateway = True
        trace_collector.mark_submitted(frame_id, frame_index, CAMERA_ID)
        trace_collector.mark_warmup(frame_id, is_warmup)
        transport.inject_message(
            _encode_frame_as_message(frame, frame_id, timestamp_ms)
        )
        queue_depth, service_state, worker_lag_ms = _capture_runtime_snapshot(service)
        record.queue_depth = queue_depth
        record.service_state = service_state
        record.worker_lag_ms = worker_lag_ms
        records.append(record)
    return records


def _count_completed_frames(service: ImageProcessingService) -> int:
    """Return processed, stale-dropped, or rejected frame count for the camera."""
    health = service.health()
    gateway_health = service._runtime.gateway.health()
    processed = int(service._metrics.get_camera_counter(CAMERA_ID, "processed_per_camera"))
    stale = int(health.stale_frames_dropped_total)
    dropped = int(health.frames_dropped_oldest_total) + int(health.frames_dropped_newest_total)
    rejected = int(gateway_health.sink_enqueue_rejected_total)
    return processed + stale + dropped + rejected


def _wait_for_phase_completion(
    service: ImageProcessingService,
    expected_completed_frames: int,
    timeout_s: float = DEFAULT_DRAIN_TIMEOUT_S,
) -> None:
    """Wait until one replay phase has drained through Gateway, IPS, and RPM."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        completed = _count_completed_frames(service)
        queue_depth, _, _ = _capture_runtime_snapshot(service)
        if completed >= expected_completed_frames and queue_depth <= 0:
            return
        time.sleep(0.05)
    raise VisualReplayError(
        "Timed out waiting for replay warmup/measurement phase to drain."
    )


def _stop_service(service: ImageProcessingService) -> None:
    """Stop the service with drain enabled."""
    service.stop(drain=True)


def _resolve_replay_mode_defaults(
    quality_mode: bool,
    output_dir: Path | None,
    replay_fps: float | None,
) -> tuple[Path, float | None]:
    """Resolve output directory and FPS defaults for the selected replay mode."""
    if not quality_mode:
        return DEFAULT_OUTPUT_DIR if output_dir is None else output_dir, replay_fps
    target_output_dir = QUALITY_OUTPUT_DIR if output_dir is None else output_dir
    target_replay_fps = QUALITY_REPLAY_FPS if replay_fps is None else replay_fps
    return target_output_dir, target_replay_fps


def _count_faces(output: dict[str, Any]) -> int:
    """Count recognized faces in one RPM output payload."""
    face_count = 0
    for person in output.get("persons", []):
        face_count += len(person.get("recognized_faces", []))
    return face_count


def _classify_missing_records(
    records: list[ReplayFrameRecord],
    stale_total: int,
    dropped_total: int,
) -> None:
    """Apply explicit stale/drop/skipped statuses to unprocessed frames."""
    stale_remaining = stale_total
    dropped_remaining = dropped_total
    for record in records:
        if record.output is not None:
            continue
        if stale_remaining > 0:
            record.status = "stale_dropped"
            record.stale_status = "true"
            stale_remaining -= 1
            continue
        if dropped_remaining > 0:
            record.status = "dropped"
            record.dropped_status = "true"
            dropped_remaining -= 1
            continue
        record.status = "skipped"


def _merge_results(
    records: list[ReplayFrameRecord],
    result_handler: RecordingResultHandler,
    trace_collector: RuntimeTraceCollector,
    gateway_health: Any,
    final_health: Any,
    service: ImageProcessingService,
    stale_total: int,
    dropped_total: int,
) -> None:
    """Attach processed outputs and explicit statuses to frame records."""
    for record in records:
        _apply_trace_defaults(record, trace_collector.get_trace_for_frame(record.frame_id))
        processed = result_handler.get_record(record.frame_id)
        if processed is None:
            continue
        output, completed_monotonic_s, rpm_frame_metrics = processed
        record.output = output
        record.processing_latency_ms = (
            completed_monotonic_s - record.submitted_monotonic_s
        ) * 1000.0
        queue_wait_ms = trace_collector.get_queue_wait_ms(record.frame_id)
        if queue_wait_ms is not None:
            record.queue_wait_ms = queue_wait_ms
        record.total_end_to_end_ms = record.processing_latency_ms
        record.persons_count = len(output.get("persons", []))
        record.faces_count = _count_faces(output)
        record.ips_processed = True
        record.ips_dequeued = True
        record.rpm_frame_metrics = rpm_frame_metrics
        _apply_rpm_metrics_to_record(record, rpm_frame_metrics)
        record.status = "processed" if record.persons_count else "no_detections"
    _classify_missing_records(records, stale_total, dropped_total)
    _finalize_ips_trace_flags(records, gateway_health, final_health, service)
    _finalize_trace_outcomes(records)
    _apply_cumulative_drop_totals(records)


def _apply_trace_defaults(record: ReplayFrameRecord, trace_row: dict[str, Any]) -> None:
    """Apply collected RPM trace defaults to one replay record."""
    record.is_warmup = bool(trace_row.get("is_warmup", False)) or record.is_warmup
    record.submitted_at_ms = int(trace_row.get("submitted_at_ms", 0))
    record.gateway_published_at_ms = int(trace_row.get("gateway_published_at_ms", 0))
    record.ips_enqueued_at_ms = int(trace_row.get("ips_enqueued_at_ms", 0))
    record.ips_dequeued_at_ms = int(trace_row.get("ips_dequeued_at_ms", 0))
    record.rpm_started_at_ms = int(trace_row.get("rpm_started_at_ms", 0))
    record.rpm_completed_at_ms = int(trace_row.get("rpm_completed_at_ms", 0))
    record.rendered_at_ms = int(trace_row.get("rendered_at_ms", 0))
    record.motion_stage_ran = bool(trace_row.get("motion_stage_ran", False))
    record.motion_detected = bool(trace_row.get("motion_detected", False))
    record.motion_regions_count = int(trace_row.get("motion_regions_count", 0))
    record.object_detection_stage_ran = bool(trace_row.get("object_detection_stage_ran", False))
    record.person_rois_count = int(trace_row.get("person_rois_count", 0))
    record.face_detection_stage_ran = bool(trace_row.get("face_detection_stage_ran", False))
    record.face_rois_count = int(trace_row.get("face_rois_count", 0))
    record.face_recognition_stage_ran = bool(trace_row.get("face_recognition_stage_ran", False))
    record.recognized_faces_count = int(trace_row.get("recognized_faces_count", 0))
    record.pipeline_error = bool(trace_row.get("pipeline_error", False))
    record.stop_reason_from_orchestrator = str(
        trace_row.get("stop_reason_from_orchestrator", "")
    )
    record.gateway_ingest_ms = _delta_ms(record.submitted_at_ms, record.gateway_published_at_ms)
    record.queue_wait_ms = _optional_float(trace_row.get("queue_wait_ms"))
    if record.queue_wait_ms is None:
        record.queue_wait_ms = _delta_ms(record.ips_enqueued_at_ms, record.rpm_started_at_ms)
    record.ips_worker_wait_ms = _delta_ms(record.ips_enqueued_at_ms, record.ips_dequeued_at_ms)
    record.worker_pre_rpm_delay_ms = _delta_ms(record.ips_dequeued_at_ms, record.rpm_started_at_ms)
    record.rpm_total_ms = _optional_float(trace_row.get("rpm_total_ms"))
    record.motion_detection_ms = _optional_float(trace_row.get("motion_detection_ms"))
    record.ftl_ingest_ms = _optional_float(trace_row.get("ftl_ingest_ms"))
    record.ftl_get_frame_total_ms = _optional_float(trace_row.get("ftl_get_frame_total_ms"))
    record.ftl_get_frame_call_count = int(trace_row.get("ftl_get_frame_call_count", 0))
    record.object_detection_total_ms = _optional_float(trace_row.get("object_detection_total_ms"))
    record.object_detection_call_count = int(trace_row.get("object_detection_call_count", 0))
    record.object_detection_avg_call_ms = _optional_float(
        trace_row.get("object_detection_avg_call_ms")
    )
    record.object_detection_max_call_ms = _optional_float(
        trace_row.get("object_detection_max_call_ms")
    )
    record.object_detection_roi_width = _optional_float(
        trace_row.get("object_detection_roi_width")
    )
    record.object_detection_roi_height = _optional_float(
        trace_row.get("object_detection_roi_height")
    )
    record.object_detection_roi_area = _optional_float(
        trace_row.get("object_detection_roi_area")
    )
    record.object_detection_largest_roi_area = int(
        trace_row.get("object_detection_largest_roi_area", 0)
    )
    record.object_detection_input_size = str(
        trace_row.get("object_detection_input_size", "unavailable")
    )
    record.object_detection_model_path = str(
        trace_row.get("object_detection_model_path", "unavailable")
    )
    record.object_detection_model_name = str(
        trace_row.get("object_detection_model_name", "unavailable")
    )
    record.object_detection_device_provider = str(
        trace_row.get("object_detection_device_provider", "unavailable")
    )
    record.object_detection_confidence_threshold = str(
        trace_row.get("object_detection_confidence_threshold", "unavailable")
    )
    record.object_detection_nms_threshold = str(
        trace_row.get("object_detection_nms_threshold", "unavailable")
    )
    record.persons_returned_per_call = str(trace_row.get("persons_returned_per_call", ""))
    record.face_detection_total_ms = _optional_float(trace_row.get("face_detection_total_ms"))
    record.face_detection_call_count = int(trace_row.get("face_detection_call_count", 0))
    record.face_detection_avg_call_ms = _optional_float(trace_row.get("face_detection_avg_call_ms"))
    record.face_detection_max_call_ms = _optional_float(trace_row.get("face_detection_max_call_ms"))
    record.face_detection_roi_width = _optional_float(trace_row.get("face_detection_roi_width"))
    record.face_detection_roi_height = _optional_float(trace_row.get("face_detection_roi_height"))
    record.face_detection_roi_area = _optional_float(trace_row.get("face_detection_roi_area"))
    record.face_detection_largest_roi_area = int(trace_row.get("face_detection_largest_roi_area", 0))
    record.face_recognition_total_ms = _optional_float(trace_row.get("face_recognition_total_ms"))
    record.face_recognition_call_count = int(trace_row.get("face_recognition_call_count", 0))
    record.person_directory_lookup_total_ms = _optional_float(
        trace_row.get("person_directory_lookup_total_ms")
    )
    record.person_directory_lookup_count = int(trace_row.get("person_directory_lookup_count", 0))
    record.output_build_ms = _optional_float(trace_row.get("output_build_ms"))
    record.motion_regions_dropped_by_cap = int(trace_row.get("motion_regions_dropped_by_cap", 0))
    record.raw_motion_regions_count = int(trace_row.get("raw_motion_regions_count", 0))
    record.filtered_motion_regions_count = int(trace_row.get("filtered_motion_regions_count", 0))
    record.merged_motion_regions_count = int(trace_row.get("merged_motion_regions_count", 0))
    record.final_motion_regions_count = int(trace_row.get("final_motion_regions_count", 0))
    record.motion_regions_dropped_by_filter = int(trace_row.get("motion_regions_dropped_by_filter", 0))
    record.motion_regions_dropped_by_merge = int(trace_row.get("motion_regions_dropped_by_merge", 0))
    record.motion_roi_areas = str(trace_row.get("motion_roi_areas", ""))
    record.motion_roi_largest_area = int(trace_row.get("motion_roi_largest_area", 0))
    record.motion_roi_smallest_area = int(trace_row.get("motion_roi_smallest_area", 0))
    record.motion_roi_total_area = int(trace_row.get("motion_roi_total_area", 0))
    record.person_rois_dropped_by_cap = int(trace_row.get("person_rois_dropped_by_cap", 0))
    record.face_rois_dropped_by_cap = int(trace_row.get("face_rois_dropped_by_cap", 0))
    record.gate_violations = str(trace_row.get("gate_violations", ""))
    record.slowest_stage = str(trace_row.get("slowest_stage", ""))
    motion_roi_bboxes = trace_row.get("motion_roi_bboxes", [])
    if isinstance(motion_roi_bboxes, list):
        record.motion_roi_bboxes = [bbox for bbox in motion_roi_bboxes if isinstance(bbox, dict)]
    record.demo_bypass_motion_gate = bool(trace_row.get("demo_bypass_motion_gate", False))


def _apply_rpm_metrics_to_record(record: ReplayFrameRecord, rpm_frame_metrics: dict[str, Any]) -> None:
    """Merge orchestrator frame metrics into a replay record trace view."""
    if not rpm_frame_metrics:
        if record.ips_processed:
            record.motion_stage_ran = True
        return
    record.is_warmup = record.is_warmup or bool(rpm_frame_metrics.get("is_warmup", False))
    total_ftl_calls = int(rpm_frame_metrics.get("total_ftl_calls_per_frame", 0))
    od_roi_requests = int(rpm_frame_metrics.get("od_roi_requests_count", 0))
    fd_roi_requests = int(rpm_frame_metrics.get("fd_roi_requests_count", 0))
    fr_roi_requests = int(rpm_frame_metrics.get("fr_roi_requests_count", 0))
    record.motion_stage_ran = bool(
        record.motion_stage_ran or rpm_frame_metrics.get("motion_stage_ran", False)
    )
    record.motion_regions_count = max(
        record.motion_regions_count,
        int(rpm_frame_metrics.get("motion_bboxes_after_merge_count", 0)),
    )
    record.motion_detected = record.motion_regions_count > 0
    record.object_detection_stage_ran = bool(record.object_detection_stage_ran or od_roi_requests > 0)
    record.person_rois_count = max(record.person_rois_count, od_roi_requests)
    record.face_detection_stage_ran = bool(record.face_detection_stage_ran or fd_roi_requests > 0)
    record.face_recognition_stage_ran = bool(record.face_recognition_stage_ran or fr_roi_requests > 0)
    record.face_rois_count = max(record.face_rois_count, fr_roi_requests)
    record.recognized_faces_count = max(record.recognized_faces_count, record.faces_count)
    record.motion_regions_dropped_by_cap = max(
        record.motion_regions_dropped_by_cap,
        int(rpm_frame_metrics.get("motion_regions_dropped_by_cap", 0)),
    )
    record.raw_motion_regions_count = max(
        record.raw_motion_regions_count,
        int(rpm_frame_metrics.get("motion_bboxes_raw_count", 0)),
    )
    record.filtered_motion_regions_count = max(
        record.filtered_motion_regions_count,
        int(rpm_frame_metrics.get("motion_bboxes_after_filter_count", 0)),
    )
    record.merged_motion_regions_count = max(
        record.merged_motion_regions_count,
        int(rpm_frame_metrics.get("motion_bboxes_after_merge_count", 0)),
    )
    record.final_motion_regions_count = max(
        record.final_motion_regions_count,
        int(rpm_frame_metrics.get("final_motion_regions_count", 0)),
    )
    record.motion_regions_dropped_by_filter = max(
        record.motion_regions_dropped_by_filter,
        int(rpm_frame_metrics.get("motion_regions_dropped_by_filter", 0)),
    )
    record.motion_regions_dropped_by_merge = max(
        record.motion_regions_dropped_by_merge,
        int(rpm_frame_metrics.get("motion_regions_dropped_by_merge", 0)),
    )
    record.motion_roi_largest_area = max(
        record.motion_roi_largest_area,
        int(rpm_frame_metrics.get("motion_roi_largest_area", 0)),
    )
    record.motion_roi_smallest_area = max(
        record.motion_roi_smallest_area,
        int(rpm_frame_metrics.get("motion_roi_smallest_area", 0)),
    )
    record.motion_roi_total_area = max(
        record.motion_roi_total_area,
        int(rpm_frame_metrics.get("motion_roi_total_area", 0)),
    )
    if not record.motion_roi_areas:
        _areas_list = rpm_frame_metrics.get("motion_roi_areas", [])
        if isinstance(_areas_list, list) and _areas_list:
            record.motion_roi_areas = _serialize_int_list(_areas_list)
    record.person_rois_dropped_by_cap = max(
        record.person_rois_dropped_by_cap,
        int(rpm_frame_metrics.get("person_rois_dropped_by_cap", 0)),
    )
    record.face_rois_dropped_by_cap = max(
        record.face_rois_dropped_by_cap,
        int(rpm_frame_metrics.get("face_rois_dropped_by_cap", 0)),
    )
    if not record.stop_reason_from_orchestrator:
        record.stop_reason_from_orchestrator = str(rpm_frame_metrics.get("stop_reason", ""))
    record.demo_bypass_motion_gate = bool(
        record.demo_bypass_motion_gate or rpm_frame_metrics.get("demo_bypass_motion_gate", False)
    )


def _optional_float(value: Any) -> float | None:
    """Convert numeric values to float or return None when unavailable."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta_ms(start_ms: int, end_ms: int) -> float | None:
    """Return end-start in ms when both timestamps are positive and ordered."""
    if start_ms <= 0 or end_ms <= 0:
        return None
    if end_ms < start_ms:
        return None
    return float(end_ms - start_ms)


def _finalize_ips_trace_flags(
    records: list[ReplayFrameRecord],
    gateway_health: Any,
    final_health: Any,
    service: ImageProcessingService,
) -> None:
    """Fill IPS ingress/queue/worker flags from final counters and statuses."""
    published_total = int(gateway_health.frames_published_total)
    enqueued_total = int(service._metrics.get_camera_counter(CAMERA_ID, "accepted_per_camera"))
    processed_total = int(service._metrics.get_camera_counter(CAMERA_ID, "processed_per_camera"))
    dequeued_total = processed_total + int(final_health.stale_frames_dropped_total)
    for record in records:
        record.submitted_to_gateway = True
        if record.output is not None:
            record.gateway_published = True
            record.ips_enqueued = True
            record.ips_dequeued = True
            continue
        if record.status == "stale_dropped":
            record.gateway_published = True
            record.ips_enqueued = True
            record.ips_dequeued = True
            continue
        if record.status == "dropped":
            record.gateway_published = True
            record.ips_enqueued = True
    _top_up_flag(records, "gateway_published", published_total)
    _top_up_flag(records, "ips_enqueued", enqueued_total)
    _top_up_flag(records, "ips_dequeued", dequeued_total)
    _top_up_flag(records, "ips_processed", processed_total)


def _top_up_flag(records: list[ReplayFrameRecord], attr_name: str, target_total: int) -> None:
    """Top up boolean flags in frame order until the requested total is reached."""
    current_total = sum(1 for record in records if bool(getattr(record, attr_name)))
    if current_total >= target_total:
        return
    for record in records:
        if current_total >= target_total:
            break
        if bool(getattr(record, attr_name)):
            continue
        setattr(record, attr_name, True)
        current_total += 1


def _finalize_trace_outcomes(records: list[ReplayFrameRecord]) -> None:
    """Resolve final runtime-trace status, stop reason, and stage path per frame."""
    for record in records:
        record.trace_status = _resolve_trace_status(record)
        record.stop_reason = _resolve_stop_reason(record)
        record.stage_path = _build_stage_path(record)


def _resolve_trace_status(record: ReplayFrameRecord) -> str:
    """Map replay status and IPS flags to the runtime-trace status vocabulary."""
    if record.pipeline_error:
        return "error"
    if record.status == "stale_dropped":
        return "stale_dropped"
    if record.status == "dropped":
        return "dropped_overflow"
    if record.status == "skipped":
        return "skipped"
    if record.output is None:
        return "no_output"
    if record.persons_count <= 0:
        return "no_output"
    return "processed"


def _resolve_stop_reason(record: ReplayFrameRecord) -> str:
    """Determine the most specific stage stop reason for one frame."""
    if record.pipeline_error:
        return "pipeline_error"
    if record.trace_status == "stale_dropped":
        return "stale_before_processing"
    if record.trace_status in {"dropped_overflow", "skipped"}:
        return "dropped_before_processing"
    if record.stop_reason_from_orchestrator:
        return record.stop_reason_from_orchestrator
    if not record.motion_stage_ran:
        return "no_motion"
    if not record.motion_detected or record.motion_regions_count <= 0:
        return "no_motion"
    if not record.object_detection_stage_ran or record.person_rois_count <= 0:
        return "no_persons"
    if not record.face_detection_stage_ran or record.face_rois_count <= 0:
        return "no_faces"
    if not record.face_recognition_stage_ran or record.recognized_faces_count <= 0:
        return "no_recognized_faces"
    return "completed"


def _build_stage_path(record: ReplayFrameRecord) -> str:
    """Build a human-readable stage path for debug overlay and CSV trace."""
    if record.trace_status in {"dropped_overflow", "stale_dropped", "skipped", "error"}:
        return f"MOTION -> STOP:{record.stop_reason}"
    if record.stop_reason == "no_motion":
        return "MOTION -> STOP:no_motion"
    if record.stop_reason == "no_persons":
        return "MOTION -> OBJECT -> STOP:no_persons"
    if record.stop_reason == "no_faces":
        return "MOTION -> OBJECT -> FACE -> STOP:no_faces"
    if record.stop_reason == "no_recognized_faces":
        return "MOTION -> OBJECT -> FACE -> RECOG -> STOP:no_recognized_faces"
    return "MOTION -> OBJECT -> FACE -> RECOG"


def _apply_cumulative_drop_totals(records: list[ReplayFrameRecord]) -> None:
    """Fill per-frame cumulative stale and drop totals for HUD rendering."""
    stale_total = 0
    dropped_total = 0
    for record in records:
        if record.status == "stale_dropped":
            stale_total += 1
        if record.status in {"dropped", "skipped"}:
            dropped_total += 1
        record.stale_frames_dropped_total = stale_total
        record.dropped_frames_total = dropped_total


def _build_overlay_frame(
    record: ReplayFrameRecord,
    debug_trace_hud: bool,
    demo_mode: bool = False,
    effective_fps: float | None = None,
) -> np.ndarray:
    """Render detections and runtime HUD for one replayed frame."""
    overlay = record.original_frame.copy()
    _draw_person_and_face_overlays(overlay, record, demo_mode)
    _draw_runtime_hud(overlay, record, debug_trace_hud, demo_mode, effective_fps)
    return overlay


def _draw_person_and_face_overlays(
    overlay: np.ndarray,
    record: ReplayFrameRecord,
    demo_mode: bool = False,
) -> None:
    """Draw person, face, and label overlays from one RPM output."""
    frame_height, frame_width = overlay.shape[:2]
    output = record.output or {}
    if demo_mode:
        for motion_bbox in record.motion_roi_bboxes:
            _draw_labeled_bbox(
                overlay,
                motion_bbox,
                frame_width,
                frame_height,
                color=MOTION_ROI_COLOR,
                thickness=MOTION_BBOX_THICKNESS,
                label="Motion ROI",
                label_color=MOTION_ROI_COLOR,
                label_scale=DEMO_FRAME_FONT_SCALE,
                label_thickness=DEMO_FRAME_FONT_THICKNESS,
                label_position="above",
            )
        if record.demo_bypass_motion_gate and not record.motion_roi_bboxes:
            _draw_labeled_bbox(
                overlay,
                {"x": 1, "y": 1, "width": frame_width - 2, "height": frame_height - 2},
                frame_width,
                frame_height,
                color=MOTION_ROI_COLOR,
                thickness=2,
                label="Demo Full Frame ROI",
                label_color=MOTION_ROI_COLOR,
                label_scale=DEMO_FRAME_FONT_SCALE,
                label_thickness=DEMO_FRAME_FONT_THICKNESS,
                label_position="above",
            )
    for person in output.get("persons", []):
        person_bbox = person.get("person_bbox")
        if person_bbox:
            _draw_labeled_bbox(
                overlay,
                person_bbox,
                frame_width,
                frame_height,
                color=PERSON_COLOR,
                thickness=PERSON_BBOX_THICKNESS if demo_mode else BORDER_THICKNESS,
                label="Person",
                label_color=PERSON_COLOR,
                label_scale=DEMO_FRAME_FONT_SCALE if demo_mode else FONT_SCALE,
                label_thickness=DEMO_FRAME_FONT_THICKNESS if demo_mode else 1,
                label_position="above",
            )
        for face in person.get("recognized_faces", []):
            face_bbox = face.get("face_bbox")
            if not face_bbox:
                continue
            _draw_labeled_bbox(
                overlay,
                face_bbox,
                frame_width,
                frame_height,
                color=FACE_COLOR,
                thickness=FACE_BBOX_THICKNESS if demo_mode else BORDER_THICKNESS,
                label="Face",
                label_color=FACE_COLOR,
                label_scale=DEMO_FRAME_FONT_SCALE if demo_mode else FONT_SCALE,
                label_thickness=DEMO_FRAME_FONT_THICKNESS if demo_mode else 1,
                label_position="below",
            )
            label = _resolve_face_label(face)
            x1, y1, x2, y2 = _clamp_bbox(face_bbox, frame_width, frame_height)
            _draw_text_with_background(
                overlay,
                label,
                (x1, y1),
                FACE_COLOR,
                DEMO_FRAME_FONT_SCALE if demo_mode else FONT_SCALE,
                RECOGNITION_LABEL_THICKNESS,
                anchor="above",
            )


def _resolve_face_label(face: dict[str, Any]) -> str:
    """Resolve the visible name label for one recognized face."""
    person_name = str(face.get("person_name", "")).strip()
    if person_name and person_name != "UNKNOWN":
        return person_name
    person_id = str(face.get("person_id", "")).strip()
    if person_id:
        return person_id
    return "UNKNOWN"


def _clamp_bbox(
    bbox: dict[str, Any],
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    """Clamp a bbox to the current frame bounds."""
    x1 = max(0, int(bbox.get("x", 0)))
    y1 = max(0, int(bbox.get("y", 0)))
    x2 = min(frame_width, x1 + int(bbox.get("width", 0)))
    y2 = min(frame_height, y1 + int(bbox.get("height", 0)))
    return x1, y1, x2, y2


def _draw_labeled_bbox(
    overlay: np.ndarray,
    bbox: dict[str, Any],
    frame_width: int,
    frame_height: int,
    color: tuple[int, int, int],
    thickness: int,
    label: str,
    label_color: tuple[int, int, int],
    label_scale: float,
    label_thickness: int,
    label_position: str,
) -> None:
    """Draw one labeled bbox and place the label where it remains readable."""
    x1, y1, x2, y2 = _clamp_bbox(bbox, frame_width, frame_height)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
    if label:
        anchor_point = (x1, y1) if label_position == "above" else (x1, y2)
        _draw_text_with_background(
            overlay,
            label,
            anchor_point,
            label_color,
            label_scale,
            label_thickness,
            anchor=label_position,
        )


def _draw_text_with_background(
    overlay: np.ndarray,
    text: str,
    anchor_point: tuple[int, int],
    text_color: tuple[int, int, int],
    font_scale: float,
    thickness: int,
    anchor: str = "above",
    background_color: tuple[int, int, int] = (0, 0, 0),
    padding: int = 4,
) -> None:
    """Draw anti-aliased text with a solid backing box for readability."""
    if not text:
        return
    (text_width, text_height), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
    x, y = anchor_point
    if anchor == "above":
        text_x = max(0, x)
        text_y = max(text_height + padding, y - padding)
        top_left = (max(0, text_x - padding), max(0, text_y - text_height - baseline - padding))
    else:
        text_x = max(0, x)
        text_y = min(overlay.shape[0] - baseline - padding, y + text_height + padding)
        top_left = (max(0, text_x - padding), max(0, text_y - text_height - baseline - padding))
    bottom_right = (
        min(overlay.shape[1] - 1, top_left[0] + text_width + (padding * 2)),
        min(overlay.shape[0] - 1, top_left[1] + text_height + baseline + (padding * 2)),
    )
    cv2.rectangle(overlay, top_left, bottom_right, background_color, -1)
    cv2.putText(
        overlay,
        text,
        (top_left[0] + padding, bottom_right[1] - baseline - padding),
        FONT,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def _draw_runtime_hud(
    overlay: np.ndarray,
    record: ReplayFrameRecord,
    debug_trace_hud: bool,
    demo_mode: bool = False,
    effective_fps: float | None = None,
) -> None:
    """Draw the runtime HUD and explicit status text for one frame."""
    if demo_mode:
        _draw_demo_hud(overlay, record, effective_fps)
        return
    hud_lines = _build_hud_lines(record, debug_trace_hud)
    for index, line in enumerate(hud_lines):
        cv2.putText(
            overlay,
            line,
            (HUD_ORIGIN_X, HUD_ORIGIN_Y + (index * HUD_LINE_HEIGHT)),
            FONT,
            FONT_SCALE,
            HUD_TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        overlay,
        f"status: {record.status}",
        (HUD_ORIGIN_X, HUD_ORIGIN_Y + (len(hud_lines) * HUD_LINE_HEIGHT)),
        FONT,
        FONT_SCALE,
        STATUS_TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )


def _draw_demo_hud(
    overlay: np.ndarray,
    record: ReplayFrameRecord,
    effective_fps: float | None,
) -> None:
    """Draw the demo HUD blocks in the upper corners."""
    frame_height, frame_width = overlay.shape[:2]
    left_lines = [
        f"Camera: {DEMO_CAMERA_LABEL}",
        f"Frame: {record.frame_index}",
        f"FPS: {effective_fps:.2f}" if effective_fps is not None else "FPS: n/a",
    ]
    right_lines = [
        f"Motion Regions: {record.motion_regions_count}",
        f"Persons: {record.persons_count}",
        f"Faces: {record.faces_count}",
        f"Recognized: {record.recognized_faces_count}",
    ]
    _draw_text_panel(overlay, left_lines, (12, 12), align_right=False)
    _draw_text_panel(overlay, right_lines, (frame_width - 12, 12), align_right=True)


def _draw_text_panel(
    overlay: np.ndarray,
    lines: list[str],
    origin: tuple[int, int],
    align_right: bool,
) -> None:
    """Draw a semi-transparent panel with stacked demo text."""
    if not lines:
        return
    font_scale = DEMO_HUD_FONT_SCALE
    thickness = DEMO_HUD_THICKNESS
    line_gap = 8
    text_sizes = [cv2.getTextSize(line, FONT, font_scale, thickness) for line in lines]
    widths = [size[0][0] for size in text_sizes]
    heights = [size[0][1] for size in text_sizes]
    baselines = [size[1] for size in text_sizes]
    panel_width = max(widths) + 28
    panel_height = sum(heights) + sum(baselines) + (line_gap * (len(lines) - 1)) + 20
    if align_right:
        x2 = origin[0]
        x1 = max(0, x2 - panel_width)
    else:
        x1 = origin[0]
        x2 = min(overlay.shape[1] - 1, x1 + panel_width)
    y1 = origin[1]
    y2 = min(overlay.shape[0] - 1, y1 + panel_height)
    _draw_translucent_panel(overlay, (x1, y1), (x2, y2), SUMMARY_BACKGROUND_COLOR, 0.58)
    cursor_y = y1 + 16
    for index, line in enumerate(lines):
        text_width = widths[index]
        text_height = heights[index]
        baseline = baselines[index]
        text_x = x2 - text_width - 14 if align_right else x1 + 14
        cv2.putText(
            overlay,
            line,
            (text_x, cursor_y + text_height),
            FONT,
            font_scale,
            SUMMARY_TEXT_COLOR,
            thickness,
            cv2.LINE_AA,
        )
        cursor_y += text_height + baseline + line_gap


def _draw_translucent_panel(
    overlay: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    """Draw a semi-transparent rectangle overlay in place."""
    x1, y1 = top_left
    x2, y2 = bottom_right
    x1 = max(0, min(x1, overlay.shape[1] - 1))
    y1 = max(0, min(y1, overlay.shape[0] - 1))
    x2 = max(x1 + 1, min(x2, overlay.shape[1] - 1))
    y2 = max(y1 + 1, min(y2, overlay.shape[0] - 1))
    roi = overlay[y1:y2, x1:x2]
    fill = np.full_like(roi, color)
    cv2.addWeighted(fill, alpha, roi, 1.0 - alpha, 0.0, roi)


def _build_summary_screen_frame(
    frame_size: tuple[int, int],
    summary: dict[str, Any],
    frame_index: int,
    summary_frame_count: int,
) -> np.ndarray:
    """Create one frame for the final summary screen."""
    frame_width, frame_height = frame_size
    frame = np.full((frame_height, frame_width, 3), SUMMARY_BACKGROUND_COLOR, dtype=np.uint8)
    _draw_translucent_panel(
        frame,
        (0, 0),
        (frame_width - 1, frame_height - 1),
        SUMMARY_BACKGROUND_COLOR,
        SUMMARY_CARD_ALPHA,
    )
    card_margin_x = max(36, frame_width // 12)
    card_margin_y = max(28, frame_height // 10)
    card_top = card_margin_y
    card_bottom = frame_height - card_margin_y
    card_left = card_margin_x
    card_right = frame_width - card_margin_x
    _draw_translucent_panel(
        frame,
        (card_left, card_top),
        (card_right, card_bottom),
        SUMMARY_CARD_COLOR,
        0.90,
    )
    _draw_centered_text(
        frame,
        "Camera Recognition Pipeline Demo",
        (frame_width // 2, card_top + 54),
        SUMMARY_TEXT_COLOR,
        SUMMARY_TITLE_FONT_SCALE,
        2,
    )
    checklist_lines = [
        ("\u2713 Motion Detection", SUMMARY_SUCCESS_COLOR),
        ("\u2713 Object Detection", SUMMARY_SUCCESS_COLOR),
        ("\u2713 Face Detection", SUMMARY_SUCCESS_COLOR),
        ("\u2713 Face Recognition", SUMMARY_SUCCESS_COLOR),
    ]
    checklist_y = card_top + 108
    for text, color in checklist_lines:
        _draw_centered_text(
            frame,
            text,
            (frame_width // 2, checklist_y),
            color,
            SUMMARY_SECTION_FONT_SCALE,
            2,
        )
        checklist_y += 34
    stats_lines = [
        f"Processed Frames: {summary.get('frames_processed', 0)}",
        f"Motion Regions Detected: {summary.get('total_motion_regions_detected', 0)}",
        f"Persons Detected: {summary.get('total_person_detections', 0)}",
        f"Faces Detected: {summary.get('total_face_detections', 0)}",
        f"Recognized Faces: {summary.get('total_recognized_faces', 0)}",
    ]
    stats_y = checklist_y + 22
    _draw_centered_text(
        frame,
        "Statistics:",
        (frame_width // 2, stats_y),
        SUMMARY_ACCENT_COLOR,
        SUMMARY_SECTION_FONT_SCALE,
        2,
    )
    stats_y += 34
    for text in stats_lines:
        _draw_centered_text(
            frame,
            text,
            (frame_width // 2, stats_y),
            SUMMARY_TEXT_COLOR,
            SUMMARY_DETAIL_FONT_SCALE,
            1,
        )
        stats_y += 28
    diagram_top = card_bottom - 145
    _draw_centered_text(
        frame,
        "Motion Detection",
        (frame_width // 2, diagram_top),
        SUMMARY_TEXT_COLOR,
        SUMMARY_DETAIL_FONT_SCALE,
        1,
    )
    _draw_centered_text(frame, "\u2193", (frame_width // 2, diagram_top + 22), SUMMARY_ACCENT_COLOR, 0.7, 1)
    _draw_centered_text(
        frame,
        "Object Detection",
        (frame_width // 2, diagram_top + 48),
        SUMMARY_TEXT_COLOR,
        SUMMARY_DETAIL_FONT_SCALE,
        1,
    )
    _draw_centered_text(frame, "\u2193", (frame_width // 2, diagram_top + 70), SUMMARY_ACCENT_COLOR, 0.7, 1)
    _draw_centered_text(
        frame,
        "Face Detection",
        (frame_width // 2, diagram_top + 96),
        SUMMARY_TEXT_COLOR,
        SUMMARY_DETAIL_FONT_SCALE,
        1,
    )
    _draw_centered_text(frame, "\u2193", (frame_width // 2, diagram_top + 118), SUMMARY_ACCENT_COLOR, 0.7, 1)
    _draw_centered_text(
        frame,
        "Face Recognition",
        (frame_width // 2, diagram_top + 144),
        SUMMARY_TEXT_COLOR,
        SUMMARY_DETAIL_FONT_SCALE,
        1,
    )
    _draw_text_with_background(
        frame,
        f"Summary {frame_index + 1}/{summary_frame_count}",
        (card_left + 18, card_bottom - 18),
        SUMMARY_ACCENT_COLOR,
        0.52,
        1,
        anchor="above",
    )
    return frame


def _draw_centered_text(
    frame: np.ndarray,
    text: str,
    center_point: tuple[int, int],
    color: tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> None:
    """Draw centered anti-aliased text."""
    (text_width, text_height), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
    center_x, center_y = center_point
    x = max(0, center_x - (text_width // 2))
    y = max(text_height + baseline, center_y)
    cv2.putText(frame, text, (x, y), FONT, font_scale, color, thickness, cv2.LINE_AA)


def _build_hud_lines(record: ReplayFrameRecord, debug_trace_hud: bool) -> list[str]:
    """Build the visible HUD strings for one frame."""
    latency_value = record.processing_latency_ms
    latency_text = "pending" if latency_value is None else f"{latency_value:.1f}"
    lines = [
        f"camera_id: {CAMERA_ID}",
        f"frame index: {record.frame_index}",
        f"frame_id: {record.frame_id}",
        f"persons: {record.persons_count}",
        f"faces: {record.faces_count}",
        f"queue depth: {record.queue_depth}",
        f"latency ms: {latency_text}",
        f"stale dropped: {record.stale_frames_dropped_total}",
        f"dropped total: {record.dropped_frames_total}",
        f"service state: {record.service_state}",
        f"worker lag ms: {record.worker_lag_ms}",
    ]
    if debug_trace_hud:
        rpm_text = "n/a" if record.rpm_total_ms is None else f"{record.rpm_total_ms:.1f}"
        queue_wait_text = "n/a" if record.queue_wait_ms is None else f"{record.queue_wait_ms:.1f}"
        ftl_ingest_text = "n/a" if record.ftl_ingest_ms is None else f"{record.ftl_ingest_ms:.1f}"
        motion_text = "n/a" if record.motion_detection_ms is None else f"{record.motion_detection_ms:.1f}"
        od_text = (
            "n/a" if record.object_detection_total_ms is None else f"{record.object_detection_total_ms:.1f}"
        )
        fd_text = (
            "n/a" if record.face_detection_total_ms is None else f"{record.face_detection_total_ms:.1f}"
        )
        fr_text = (
            "n/a"
            if record.face_recognition_total_ms is None
            else f"{record.face_recognition_total_ms:.1f}"
        )
        pd_text = (
            "n/a"
            if record.person_directory_lookup_total_ms is None
            else f"{record.person_directory_lookup_total_ms:.1f}"
        )
        lines.extend(
            [
                f"stage path: {record.stage_path}",
                f"rpm total ms: {rpm_text}",
                f"queue wait ms: {queue_wait_text}",
                f"slowest stage: {record.slowest_stage or 'n/a'}",
                f"ftl ingest ms: {ftl_ingest_text}",
                f"motion ms: {motion_text}",
                f"od total ms: {od_text}",
                f"fd total ms: {fd_text}",
                f"fr total ms: {fr_text}",
                f"person dir ms: {pd_text}",
                f"stop reason: {record.stop_reason}",
            ]
        )
    return lines


def _compose_side_by_side(
    original_frame: np.ndarray,
    overlay_frame: np.ndarray,
) -> np.ndarray:
    """Compose the optional side-by-side presentation frame."""
    original = original_frame.copy()
    overlay = overlay_frame.copy()
    cv2.putText(
        original,
        "ORIGINAL",
        (HUD_ORIGIN_X, HUD_ORIGIN_Y),
        FONT,
        FONT_SCALE,
        PANEL_LABEL_COLOR,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        "RUNTIME OVERLAY",
        (HUD_ORIGIN_X, HUD_ORIGIN_Y),
        FONT,
        FONT_SCALE,
        PANEL_LABEL_COLOR,
        1,
        cv2.LINE_AA,
    )
    return np.concatenate([original, overlay], axis=1)


def _open_video_writer(
    output_path: Path,
    fps: float,
    frame_size: tuple[int, int],
) -> cv2.VideoWriter:
    """Open an mp4 writer or raise a replay error."""
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*FOURCC_MP4V),  # type: ignore[attr-defined]
        fps,
        frame_size,
    )
    if writer.isOpened():
        return writer
    raise VisualReplayError(f"Unable to open output video writer: {output_path}")


def _render_videos(
    records: list[ReplayFrameRecord],
    output_dir: Path,
    fps: float,
    enable_side_by_side: bool,
    sample_frame_interval: int,
    debug_trace_hud: bool,
    demo_mode: bool = False,
    summary_screen_seconds: float = 0.0,
    overlay_video_name: str = OVERLAY_VIDEO_NAME,
) -> tuple[Path, Path | None, Path | None, int]:
    """Render overlay and side-by-side videos from merged frame records."""
    overlay_path = output_dir / overlay_video_name
    side_by_side_path = output_dir / SIDE_BY_SIDE_VIDEO_NAME
    sample_frames_dir: Path | None = None
    height, width = records[0].original_frame.shape[:2]
    overlay_writer = _open_video_writer(overlay_path, fps, (width, height))
    side_by_side_writer = None
    if enable_side_by_side:
        side_by_side_writer = _open_video_writer(side_by_side_path, fps, (width * 2, height))
    if sample_frame_interval > 0:
        sample_frames_dir = output_dir / SAMPLE_FRAMES_DIR_NAME
        sample_frames_dir.mkdir(parents=True, exist_ok=True)
    summary_frame_count = 0
    try:
        for record in records:
            overlay_frame = _build_overlay_frame(record, debug_trace_hud, demo_mode, fps)
            overlay_writer.write(overlay_frame)
            record.rendered_at_ms = int(time.monotonic() * 1000)
            end_to_end_ms = _delta_ms(record.submitted_at_ms, record.rendered_at_ms)
            if end_to_end_ms is not None:
                record.total_end_to_end_ms = end_to_end_ms
            if side_by_side_writer is not None:
                side_by_side_writer.write(
                    _compose_side_by_side(record.original_frame, overlay_frame)
                )
            _write_sample_frame(sample_frames_dir, sample_frame_interval, record, overlay_frame)
        if demo_mode and summary_screen_seconds > 0.0:
            summary_frame_count = max(1, int(round(summary_screen_seconds * fps)))
            summary_payload = _build_demo_summary_payload(records, fps)
            for index in range(summary_frame_count):
                summary_frame = _build_summary_screen_frame(
                    (width, height),
                    summary_payload,
                    index,
                    summary_frame_count,
                )
                overlay_writer.write(summary_frame)
                if side_by_side_writer is not None:
                    side_by_side_writer.write(_compose_side_by_side(summary_frame, summary_frame))
    finally:
        overlay_writer.release()
        if side_by_side_writer is not None:
            side_by_side_writer.release()
    return (
        overlay_path,
        side_by_side_path if enable_side_by_side else None,
        sample_frames_dir,
        summary_frame_count,
    )


def _write_sample_frame(
    sample_frames_dir: Path | None,
    sample_frame_interval: int,
    record: ReplayFrameRecord,
    overlay_frame: np.ndarray,
) -> None:
    """Optionally persist one PNG sample frame at the requested interval."""
    if sample_frames_dir is None or sample_frame_interval <= 0:
        return
    if record.frame_index % sample_frame_interval != 0:
        return
    target_path = sample_frames_dir / f"{SAMPLE_FRAME_STEM}_{record.frame_index:06d}.png"
    cv2.imwrite(str(target_path), overlay_frame)


def _build_demo_summary_payload(records: list[ReplayFrameRecord], fps: float) -> dict[str, Any]:
    """Build the counts needed for the final demo summary card."""
    return {
        "frames_processed": sum(1 for record in records if record.output is not None),
        "total_motion_regions_detected": sum(
            max(len(record.motion_roi_bboxes), record.motion_regions_count) for record in records
        ),
        "total_person_detections": sum(record.persons_count for record in records),
        "total_face_detections": sum(record.faces_count for record in records),
        "total_recognized_faces": sum(record.recognized_faces_count for record in records),
        "average_fps": float(fps),
    }


def _write_metrics_csv(output_dir: Path, records: list[ReplayFrameRecord]) -> Path:
    """Write one CSV row per rendered frame."""
    metrics_path = output_dir / METRICS_NAME
    with open(metrics_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_metrics_fieldnames())
        writer.writeheader()
        for record in records:
            writer.writerow(_record_to_metrics_row(record))
    return metrics_path


def _metrics_fieldnames() -> list[str]:
    """Return the CSV column order for replay metrics."""
    return [
        "frame_id",
        "frame_index",
        "camera_id",
        "timestamp_ms",
        "queue_depth",
        "processing_latency_ms",
        "stale_status",
        "dropped_status",
        "persons_count",
        "faces_count",
        "service_state",
        "status",
    ]


def _record_to_metrics_row(record: ReplayFrameRecord) -> dict[str, Any]:
    """Convert one frame record into a CSV row."""
    return {
        "frame_id": record.frame_id,
        "frame_index": record.frame_index,
        "camera_id": CAMERA_ID,
        "timestamp_ms": record.timestamp_ms,
        "queue_depth": record.queue_depth,
        "processing_latency_ms": ""
        if record.processing_latency_ms is None
        else f"{record.processing_latency_ms:.3f}",
        "stale_status": record.stale_status,
        "dropped_status": record.dropped_status,
        "persons_count": record.persons_count,
        "faces_count": record.faces_count,
        "service_state": record.service_state,
        "status": record.status,
    }


def _write_runtime_trace_csv(output_dir: Path, records: list[ReplayFrameRecord]) -> Path:
    """Write frame-by-frame IPS and RPM stage trace diagnostics."""
    trace_path = output_dir / RUNTIME_TRACE_NAME
    with open(trace_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_runtime_trace_fieldnames())
        writer.writeheader()
        for record in records:
            writer.writerow(_record_to_runtime_trace_row(record))
    return trace_path


def _runtime_trace_fieldnames() -> list[str]:
    """Return column order for runtime_trace.csv."""
    return [
        "frame_id",
        "frame_index",
        "camera_id",
        "is_warmup",
        "submitted_to_gateway",
        "submitted_at_ms",
        "gateway_published",
        "gateway_published_at_ms",
        "ips_enqueued",
        "ips_enqueued_at_ms",
        "ips_dequeued",
        "ips_dequeued_at_ms",
        "ips_processed",
        "rpm_started_at_ms",
        "rpm_completed_at_ms",
        "rendered_at_ms",
        "status",
        "motion_stage_ran",
        "motion_detected",
        "demo_bypass_motion_gate",
        "motion_regions_count",
        "motion_regions_dropped_by_cap",
        "raw_motion_regions_count",
        "filtered_motion_regions_count",
        "merged_motion_regions_count",
        "final_motion_regions_count",
        "motion_regions_dropped_by_filter",
        "motion_regions_dropped_by_merge",
        "motion_roi_areas",
        "motion_roi_largest_area",
        "motion_roi_smallest_area",
        "motion_roi_total_area",
        "object_detection_stage_ran",
        "person_rois_count",
        "person_rois_dropped_by_cap",
        "face_detection_stage_ran",
        "face_rois_count",
        "face_rois_dropped_by_cap",
        "face_recognition_stage_ran",
        "recognized_faces_count",
        "persons_count",
        "faces_count",
        "stop_reason",
        "stop_reason_from_orchestrator",
        "stage_path",
        "gate_violations",
        "slowest_stage",
        "gateway_ingest_ms",
        "queue_wait_ms",
        "ips_worker_wait_ms",
        "worker_pre_rpm_delay_ms",
        "rpm_total_ms",
        "motion_detection_ms",
        "ftl_ingest_ms",
        "ftl_get_frame_total_ms",
        "ftl_get_frame_call_count",
        "object_detection_total_ms",
        "object_detection_call_count",
        "object_detection_avg_call_ms",
        "object_detection_max_call_ms",
        "object_detection_roi_width",
        "object_detection_roi_height",
        "object_detection_roi_area",
        "object_detection_largest_roi_area",
        "object_detection_input_size",
        "object_detection_model_path",
        "object_detection_model_name",
        "object_detection_device_provider",
        "object_detection_confidence_threshold",
        "object_detection_nms_threshold",
        "persons_returned_per_call",
        "face_detection_total_ms",
        "face_detection_call_count",
        "face_detection_avg_call_ms",
        "face_detection_max_call_ms",
        "face_detection_roi_width",
        "face_detection_roi_height",
        "face_detection_roi_area",
        "face_detection_largest_roi_area",
        "face_recognition_total_ms",
        "face_recognition_call_count",
        "person_directory_lookup_total_ms",
        "person_directory_lookup_count",
        "output_build_ms",
        "total_end_to_end_ms",
    ]


def _record_to_runtime_trace_row(record: ReplayFrameRecord) -> dict[str, Any]:
    """Convert one replay record into a runtime-trace CSV row."""
    return {
        "frame_id": record.frame_id,
        "frame_index": record.frame_index,
        "camera_id": CAMERA_ID,
        "is_warmup": record.is_warmup,
        "submitted_to_gateway": record.submitted_to_gateway,
        "submitted_at_ms": record.submitted_at_ms,
        "gateway_published": record.gateway_published,
        "gateway_published_at_ms": record.gateway_published_at_ms,
        "ips_enqueued": record.ips_enqueued,
        "ips_enqueued_at_ms": record.ips_enqueued_at_ms,
        "ips_dequeued": record.ips_dequeued,
        "ips_dequeued_at_ms": record.ips_dequeued_at_ms,
        "ips_processed": record.ips_processed,
        "rpm_started_at_ms": record.rpm_started_at_ms,
        "rpm_completed_at_ms": record.rpm_completed_at_ms,
        "rendered_at_ms": record.rendered_at_ms,
        "status": record.trace_status,
        "motion_stage_ran": record.motion_stage_ran,
        "motion_detected": record.motion_detected,
        "demo_bypass_motion_gate": record.demo_bypass_motion_gate,
        "motion_regions_count": record.motion_regions_count,
        "motion_regions_dropped_by_cap": record.motion_regions_dropped_by_cap,
        "raw_motion_regions_count": record.raw_motion_regions_count,
        "filtered_motion_regions_count": record.filtered_motion_regions_count,
        "merged_motion_regions_count": record.merged_motion_regions_count,
        "final_motion_regions_count": record.final_motion_regions_count,
        "motion_regions_dropped_by_filter": record.motion_regions_dropped_by_filter,
        "motion_regions_dropped_by_merge": record.motion_regions_dropped_by_merge,
        "motion_roi_areas": record.motion_roi_areas,
        "motion_roi_largest_area": record.motion_roi_largest_area,
        "motion_roi_smallest_area": record.motion_roi_smallest_area,
        "motion_roi_total_area": record.motion_roi_total_area,
        "object_detection_stage_ran": record.object_detection_stage_ran,
        "person_rois_count": record.person_rois_count,
        "person_rois_dropped_by_cap": record.person_rois_dropped_by_cap,
        "face_detection_stage_ran": record.face_detection_stage_ran,
        "face_rois_count": record.face_rois_count,
        "face_rois_dropped_by_cap": record.face_rois_dropped_by_cap,
        "face_recognition_stage_ran": record.face_recognition_stage_ran,
        "recognized_faces_count": record.recognized_faces_count,
        "persons_count": record.persons_count,
        "faces_count": record.faces_count,
        "stop_reason": record.stop_reason,
        "stop_reason_from_orchestrator": record.stop_reason_from_orchestrator,
        "stage_path": record.stage_path,
        "gate_violations": record.gate_violations,
        "slowest_stage": record.slowest_stage,
        "gateway_ingest_ms": _csv_float(record.gateway_ingest_ms),
        "queue_wait_ms": _csv_float(record.queue_wait_ms),
        "ips_worker_wait_ms": _csv_float(record.ips_worker_wait_ms),
        "worker_pre_rpm_delay_ms": _csv_float(record.worker_pre_rpm_delay_ms),
        "rpm_total_ms": _csv_float(record.rpm_total_ms),
        "motion_detection_ms": _csv_float(record.motion_detection_ms),
        "ftl_ingest_ms": _csv_float(record.ftl_ingest_ms),
        "ftl_get_frame_total_ms": _csv_float(record.ftl_get_frame_total_ms),
        "ftl_get_frame_call_count": record.ftl_get_frame_call_count,
        "object_detection_total_ms": _csv_float(record.object_detection_total_ms),
        "object_detection_call_count": record.object_detection_call_count,
        "object_detection_avg_call_ms": _csv_float(record.object_detection_avg_call_ms),
        "object_detection_max_call_ms": _csv_float(record.object_detection_max_call_ms),
        "object_detection_roi_width": _csv_float(record.object_detection_roi_width),
        "object_detection_roi_height": _csv_float(record.object_detection_roi_height),
        "object_detection_roi_area": _csv_float(record.object_detection_roi_area),
        "object_detection_largest_roi_area": record.object_detection_largest_roi_area,
        "object_detection_input_size": record.object_detection_input_size,
        "object_detection_model_path": record.object_detection_model_path,
        "object_detection_model_name": record.object_detection_model_name,
        "object_detection_device_provider": record.object_detection_device_provider,
        "object_detection_confidence_threshold": record.object_detection_confidence_threshold,
        "object_detection_nms_threshold": record.object_detection_nms_threshold,
        "persons_returned_per_call": record.persons_returned_per_call,
        "face_detection_total_ms": _csv_float(record.face_detection_total_ms),
        "face_detection_call_count": record.face_detection_call_count,
        "face_detection_avg_call_ms": _csv_float(record.face_detection_avg_call_ms),
        "face_detection_max_call_ms": _csv_float(record.face_detection_max_call_ms),
        "face_detection_roi_width": _csv_float(record.face_detection_roi_width),
        "face_detection_roi_height": _csv_float(record.face_detection_roi_height),
        "face_detection_roi_area": _csv_float(record.face_detection_roi_area),
        "face_detection_largest_roi_area": record.face_detection_largest_roi_area,
        "face_recognition_total_ms": _csv_float(record.face_recognition_total_ms),
        "face_recognition_call_count": record.face_recognition_call_count,
        "person_directory_lookup_total_ms": _csv_float(record.person_directory_lookup_total_ms),
        "person_directory_lookup_count": record.person_directory_lookup_count,
        "output_build_ms": _csv_float(record.output_build_ms),
        "total_end_to_end_ms": _csv_float(record.total_end_to_end_ms),
    }


def _csv_float(value: float | None) -> str:
    """Render optional float values for CSV output."""
    if value is None:
        return ""
    return f"{value:.3f}"


def _serialize_int_list(values: Any) -> str:
    """Serialize integer list values for CSV output columns."""
    if not isinstance(values, list):
        return ""
    normalized = [int(v) for v in values]
    return json.dumps(normalized, separators=(",", ":"))


def _deserialize_int_list(value: str) -> list[int]:
    """Parse JSON-encoded int list and fallback to empty list on errors."""
    if not value:
        return []
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(loaded, list):
        return []
    return [int(v) for v in loaded]


def _pearson_correlation(xs: list[float], ys: list[float]) -> float:
    """Return Pearson correlation for aligned vectors or 0.0 when undefined."""
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = _average(xs)
    mean_y = _average(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0.0 or var_y <= 0.0:
        return 0.0
    return float(cov / ((var_x ** 0.5) * (var_y ** 0.5)))


def _build_runtime_trace_summary(records: list[ReplayFrameRecord]) -> dict[str, Any]:
    """Build aggregated runtime trace counters and latency summaries."""
    warmup_records = [record for record in records if _is_warmup_record(record)]
    measured_records = [record for record in records if not _is_warmup_record(record)]
    records_for_summary = measured_records if measured_records else records
    stop_reason_counts: dict[str, int] = {}
    for record in records_for_summary:
        stop_reason_counts[record.stop_reason] = stop_reason_counts.get(record.stop_reason, 0) + 1
    latency_keys = [
        "gateway_ingest_ms",
        "queue_wait_ms",
        "ips_worker_wait_ms",
        "worker_pre_rpm_delay_ms",
        "rpm_total_ms",
        "motion_detection_ms",
        "ftl_ingest_ms",
        "ftl_get_frame_total_ms",
        "object_detection_total_ms",
        "face_detection_total_ms",
        "face_recognition_total_ms",
        "person_directory_lookup_total_ms",
        "output_build_ms",
        "total_end_to_end_ms",
    ]
    averages: dict[str, float] = {}
    maximums: dict[str, float] = {}
    minimums: dict[str, float] = {}
    for key in latency_keys:
        values = [
            float(v)
            for v in (_latency_value(record, key) for record in records_for_summary)
            if v is not None
        ]
        averages[key] = _average(values)
        maximums[key] = _maximum(values)
        minimums[key] = min(values) if values else 0.0
    frames_with_four = [
        record.frame_index for record in records_for_summary if record.persons_count >= 4
    ]
    # throughput estimates
    rpm_times = [
        r.rpm_total_ms
        for r in records_for_summary
        if r.rpm_total_ms is not None and r.rpm_total_ms > 0
    ]
    avg_rpm_ms = _average(rpm_times)
    estimated_rpm_fps = (1000.0 / avg_rpm_ms) if avg_rpm_ms > 0 else 0.0
    gate_violation_count = sum(
        1 for r in records_for_summary if r.gate_violations and r.gate_violations.strip()
    )
    # estimate input FPS from total frames and total wall time if available
    e2e_times = [
        r.total_end_to_end_ms
        for r in records_for_summary
        if r.total_end_to_end_ms is not None and r.total_end_to_end_ms > 0
    ]
    avg_e2e_ms = _average(e2e_times)
    estimated_input_fps = (1000.0 / avg_e2e_ms) if avg_e2e_ms > 0 else 0.0
    throughput_ratio = (estimated_rpm_fps / estimated_input_fps) if estimated_input_fps > 0 else 0.0
    processed_records = [record for record in records_for_summary if record.ips_processed]
    od_records = [record for record in processed_records if record.object_detection_call_count > 0]
    fd_records = [record for record in processed_records if record.face_detection_call_count > 0]
    fr_records = [record for record in processed_records if record.face_recognition_call_count > 0]
    total_od_calls = sum(record.object_detection_call_count for record in od_records)
    total_od_ms = sum((record.object_detection_total_ms or 0.0) for record in od_records)
    total_fd_calls = sum(record.face_detection_call_count for record in fd_records)
    total_fd_ms = sum((record.face_detection_total_ms or 0.0) for record in fd_records)
    total_fr_calls = sum(record.face_recognition_call_count for record in fr_records)
    total_fr_ms = sum((record.face_recognition_total_ms or 0.0) for record in fr_records)
    avg_od_calls = _average([float(record.object_detection_call_count) for record in processed_records])
    max_od_calls = max((record.object_detection_call_count for record in records_for_summary), default=0)
    avg_fd_calls = _average([float(record.face_detection_call_count) for record in processed_records])
    max_fd_calls = max((record.face_detection_call_count for record in records_for_summary), default=0)
    avg_od_roi_area = _average([
        float(record.object_detection_roi_area or 0.0)
        for record in od_records
        if record.object_detection_roi_area is not None
    ])
    avg_fd_roi_area = _average([
        float(record.face_detection_roi_area or 0.0)
        for record in fd_records
        if record.face_detection_roi_area is not None
    ])
    largest_od_roi_area = max(
        (record.object_detection_largest_roi_area for record in records_for_summary),
        default=0,
    )
    largest_fd_roi_area = max(
        (record.face_detection_largest_roi_area for record in records_for_summary),
        default=0,
    )
    slowest_od_record = max(
        records_for_summary,
        key=lambda record: float(record.object_detection_total_ms or 0.0),
        default=None,
    )
    slowest_fd_record = max(
        records_for_summary,
        key=lambda record: float(record.face_detection_total_ms or 0.0),
        default=None,
    )
    roi_count_series = [float(record.object_detection_call_count) for record in od_records]
    roi_area_series = [float(record.object_detection_roi_area or 0.0) for record in od_records]
    od_latency_series = [float(record.object_detection_total_ms or 0.0) for record in od_records]
    corr_roi_count_vs_od_latency = _pearson_correlation(roi_count_series, od_latency_series)
    corr_roi_area_vs_od_latency = _pearson_correlation(roi_area_series, od_latency_series)
    fd_person_roi_series = [float(record.person_rois_count) for record in fd_records]
    fd_latency_series = [float(record.face_detection_total_ms or 0.0) for record in fd_records]
    fd_roi_area_series = [float(record.face_detection_roi_area or 0.0) for record in fd_records]
    corr_person_rois_vs_fd_latency = _pearson_correlation(fd_person_roi_series, fd_latency_series)
    corr_fd_roi_area_vs_fd_latency = _pearson_correlation(fd_roi_area_series, fd_latency_series)
    persons_per_call_values = [
        value
        for record in od_records
        for value in _deserialize_int_list(record.persons_returned_per_call)
    ]
    metadata_source = od_records[0] if od_records else (records_for_summary[0] if records_for_summary else None)
    backend_diagnostics = _build_backend_diagnostics(records)
    warmup_first_frame_rpm_total_ms = _first_positive_latency(warmup_records, "rpm_total_ms")
    warmup_max_rpm_total_ms = _maximum([
        float(record.rpm_total_ms)
        for record in warmup_records
        if record.rpm_total_ms is not None
    ])
    first_measured_frame_rpm_total_ms = _first_positive_latency(
        records_for_summary,
        "rpm_total_ms",
    )
    return {
        "total_frames_submitted": len(records_for_summary),
        "warmup_frames_submitted": len(warmup_records),
        "warmup_frames_processed": sum(1 for r in warmup_records if r.ips_processed),
        "total_gateway_published": sum(1 for r in records_for_summary if r.gateway_published),
        "total_ips_enqueued": sum(1 for r in records_for_summary if r.ips_enqueued),
        "total_ips_dequeued": sum(1 for r in records_for_summary if r.ips_dequeued),
        "total_ips_processed": sum(1 for r in records_for_summary if r.ips_processed),
        "total_dropped_overflow": sum(
            1 for r in records_for_summary if r.trace_status == "dropped_overflow"
        ),
        "total_stale_dropped": sum(
            1 for r in records_for_summary if r.trace_status == "stale_dropped"
        ),
        "total_motion_stage_ran": sum(1 for r in records_for_summary if r.motion_stage_ran),
        "total_motion_detected": sum(1 for r in records_for_summary if r.motion_detected),
        "total_demo_bypass_frames": sum(
            1 for r in records_for_summary if r.demo_bypass_motion_gate
        ),
        "total_object_detection_stage_ran": sum(
            1 for r in records_for_summary if r.object_detection_stage_ran
        ),
        "total_face_detection_stage_ran": sum(
            1 for r in records_for_summary if r.face_detection_stage_ran
        ),
        "total_face_recognition_stage_ran": sum(
            1 for r in records_for_summary if r.face_recognition_stage_ran
        ),
        "total_motion_regions_dropped_by_cap": sum(
            int(r.motion_regions_dropped_by_cap) for r in records_for_summary
        ),
        "total_person_rois_dropped_by_cap": sum(
            int(r.person_rois_dropped_by_cap) for r in records_for_summary
        ),
        "total_face_rois_dropped_by_cap": sum(
            int(r.face_rois_dropped_by_cap) for r in records_for_summary
        ),
        "stop_reason_counts": stop_reason_counts,
        "average_latency_ms_by_stage": averages,
        "max_latency_ms_by_stage": maximums,
        "min_latency_ms_by_stage": minimums,
        "max_motion_regions_count": max(
            (r.motion_regions_count for r in records_for_summary), default=0
        ),
        "average_raw_motion_regions_per_frame": _average(
            [float(r.raw_motion_regions_count) for r in processed_records]
        ),
        "average_final_motion_regions_per_frame": _average(
            [float(r.final_motion_regions_count) for r in processed_records]
        ),
        "max_raw_motion_regions": max(
            (r.raw_motion_regions_count for r in records_for_summary), default=0
        ),
        "max_final_motion_regions": max(
            (r.final_motion_regions_count for r in records_for_summary), default=0
        ),
        "average_motion_regions_dropped_by_filter": _average(
            [float(r.motion_regions_dropped_by_filter) for r in processed_records]
        ),
        "average_motion_regions_dropped_by_merge": _average(
            [float(r.motion_regions_dropped_by_merge) for r in processed_records]
        ),
        "average_motion_regions_dropped_by_cap": _average(
            [float(r.motion_regions_dropped_by_cap) for r in processed_records]
        ),
        "correlation_final_motion_roi_count_vs_od_latency": _pearson_correlation(
            [float(r.final_motion_regions_count) for r in od_records],
            od_latency_series,
        ),
        "correlation_motion_roi_total_area_vs_od_latency": _pearson_correlation(
            [float(r.motion_roi_total_area) for r in od_records],
            od_latency_series,
        ),
        "top10_frames_by_motion_roi_count": [
            r.frame_index
            for r in sorted(
                records_for_summary,
                key=lambda r: r.final_motion_regions_count,
                reverse=True,
            )[:10]
            if r.final_motion_regions_count > 0
        ],
        "top10_frames_by_od_latency": [
            r.frame_index
            for r in sorted(
                records_for_summary,
                key=lambda r: float(r.object_detection_total_ms or 0.0),
                reverse=True,
            )[:10]
            if (r.object_detection_total_ms or 0.0) > 0.0
        ],
        "max_person_rois_count": max((r.person_rois_count for r in records_for_summary), default=0),
        "max_face_rois_count": max((r.face_rois_count for r in records_for_summary), default=0),
        "frame_indexes_with_4_persons": frames_with_four,
        "bottleneck_stage_guess": _guess_bottleneck_stage(maximums),
        "estimated_input_fps": round(estimated_input_fps, 2),
        "estimated_rpm_fps": round(estimated_rpm_fps, 2),
        "throughput_ratio": round(throughput_ratio, 4),
        "gate_violation_count": gate_violation_count,
        "first_measured_frame_rpm_total_ms": first_measured_frame_rpm_total_ms,
        "warmup_section": {
            "warmup_frames_submitted": len(warmup_records),
            "warmup_frames_processed": sum(1 for r in warmup_records if r.ips_processed),
            "warmup_first_frame_rpm_total_ms": warmup_first_frame_rpm_total_ms,
            "warmup_max_rpm_total_ms": warmup_max_rpm_total_ms,
        },
        "average_od_calls_per_processed_frame": avg_od_calls,
        "max_od_calls_per_frame": int(max_od_calls),
        "average_od_roi_area": avg_od_roi_area,
        "largest_od_roi_area": int(largest_od_roi_area),
        "average_od_latency_per_call_ms": _safe_divide(total_od_ms, float(total_od_calls)),
        "average_object_detection_model_ms_over_calls": _safe_divide(
            total_od_ms,
            float(total_od_calls),
        ),
        "average_fd_calls_per_processed_frame": avg_fd_calls,
        "max_fd_calls_per_frame": int(max_fd_calls),
        "average_fd_roi_area": avg_fd_roi_area,
        "largest_fd_roi_area": int(largest_fd_roi_area),
        "average_fd_latency_per_call_ms": _safe_divide(total_fd_ms, float(total_fd_calls)),
        "average_face_detection_model_ms_over_calls": _safe_divide(
            total_fd_ms,
            float(total_fd_calls),
        ),
        "average_face_recognition_model_ms_over_calls": _safe_divide(
            total_fr_ms,
            float(total_fr_calls),
        ),
        "face_detection_avg_call_ms": _safe_divide(total_fd_ms, float(total_fd_calls)),
        "face_detection_max_call_ms": max(
            (float(record.face_detection_max_call_ms or 0.0) for record in fd_records),
            default=0.0,
        ),
        "slowest_od_frame_id": "" if slowest_od_record is None else slowest_od_record.frame_id,
        "slowest_od_frame_index": -1 if slowest_od_record is None else slowest_od_record.frame_index,
        "slowest_fd_frame_id": "" if slowest_fd_record is None else slowest_fd_record.frame_id,
        "slowest_fd_frame_index": -1 if slowest_fd_record is None else slowest_fd_record.frame_index,
        "correlation_roi_count_vs_od_latency": corr_roi_count_vs_od_latency,
        "correlation_roi_area_vs_od_latency": corr_roi_area_vs_od_latency,
        "correlation_person_rois_vs_fd_latency": corr_person_rois_vs_fd_latency,
        "correlation_fd_roi_area_vs_fd_latency": corr_fd_roi_area_vs_fd_latency,
        "average_persons_returned_per_od_call": _average(
            [float(value) for value in persons_per_call_values]
        ),
        "od_metadata": {
            "object_detection_input_size": "unavailable"
            if metadata_source is None else metadata_source.object_detection_input_size,
            "object_detection_model_path": "unavailable"
            if metadata_source is None else metadata_source.object_detection_model_path,
            "object_detection_model_name": "unavailable"
            if metadata_source is None else metadata_source.object_detection_model_name,
            "object_detection_device_provider": "unavailable"
            if metadata_source is None else metadata_source.object_detection_device_provider,
            "object_detection_confidence_threshold": "unavailable"
            if metadata_source is None else metadata_source.object_detection_confidence_threshold,
            "object_detection_nms_threshold": "unavailable"
            if metadata_source is None else metadata_source.object_detection_nms_threshold,
        },
        "backend_diagnostics": backend_diagnostics,
    }


def _is_warmup_record(record: ReplayFrameRecord) -> bool:
    """Return whether a replay record belongs to the warmup phase."""
    if record.is_warmup:
        return True
    if not record.rpm_frame_metrics:
        return False
    return bool(record.rpm_frame_metrics.get("is_warmup", False))


def _first_positive_latency(records: list[ReplayFrameRecord], key: str) -> float:
    """Return the first positive latency value for the given record field."""
    for record in sorted(records, key=lambda item: item.frame_index):
        value = _latency_value(record, key)
        if value is not None and value > 0.0:
            return float(value)
    return 0.0


def _build_backend_diagnostics(records: list[ReplayFrameRecord]) -> dict[str, dict[str, str]]:
    """Build per-component backend diagnostics from collected replay records."""
    source = _first_record_with_backend_metadata(records)
    od_backend, od_device = _split_backend_device(
        "" if source is None else source.object_detection_device_provider
    )
    metrics = {} if source is None or source.rpm_frame_metrics is None else source.rpm_frame_metrics
    return {
        "object_detection": {
            "model_path": "unknown" if source is None else source.object_detection_model_path,
            "model_name": "unknown" if source is None else source.object_detection_model_name,
            "backend": od_backend,
            "device_provider": od_device,
            "input_size": "unknown" if source is None else source.object_detection_input_size,
            "confidence_threshold": "unknown"
            if source is None
            else source.object_detection_confidence_threshold,
            "nms_threshold": "unknown"
            if source is None
            else source.object_detection_nms_threshold,
            "cuda_available": _string_metric(metrics, "object_detection_cuda_available"),
            "inference_device": _string_metric(metrics, "object_detection_inference_device"),
            "why_unknown": _string_metric(metrics, "object_detection_why_unknown"),
        },
        "face_detection": {
            "model_path": _string_metric(metrics, "face_detection_model_path"),
            "model_name": _string_metric(metrics, "face_detection_model_name"),
            "backend": _string_metric(metrics, "face_detection_backend"),
            "device_provider": _string_metric(metrics, "face_detection_device_provider"),
            "input_size": _string_metric(metrics, "face_detection_input_size"),
            "confidence_threshold": _string_metric(metrics, "face_detection_confidence_threshold"),
        },
        "face_recognition": {
            "model_path": _string_metric(metrics, "face_recognition_model_path"),
            "model_name": _string_metric(metrics, "face_recognition_model_name"),
            "backend": _string_metric(metrics, "face_recognition_backend"),
            "device_provider": _string_metric(metrics, "face_recognition_device_provider"),
            "embedding_model": _string_metric(metrics, "face_recognition_embedding_model"),
        },
        "motion_detection": {
            "backend": _string_metric(metrics, "motion_detection_backend"),
            "device_provider": _string_metric(metrics, "motion_detection_device_provider"),
        },
    }


def _first_record_with_backend_metadata(
    records: list[ReplayFrameRecord],
) -> ReplayFrameRecord | None:
    """Return the first record carrying any backend metadata."""
    for record in sorted(records, key=lambda item: item.frame_index):
        if record.object_detection_model_path != "unavailable":
            return record
        if record.rpm_frame_metrics:
            return record
    return None


def _split_backend_device(value: str) -> tuple[str, str]:
    """Split a combined backend/device string into stable fields."""
    raw_value = value.strip()
    if not raw_value or raw_value == "unavailable":
        return "unknown", "unknown"
    if ":" not in raw_value:
        return raw_value, "unknown"
    backend, device = raw_value.split(":", 1)
    return backend or "unknown", device or "unknown"


def _string_metric(metrics: dict[str, Any], key: str) -> str:
    """Read one string diagnostic metric with an unknown fallback."""
    value = metrics.get(key, "unknown")
    text = str(value).strip()
    return text if text else "unknown"


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return safe division with zero fallback."""
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _latency_value(record: ReplayFrameRecord, key: str) -> float | None:
    """Read one latency field by name from ReplayFrameRecord."""
    value = getattr(record, key)
    return None if value is None else float(value)


def _guess_bottleneck_stage(maximums: dict[str, float]) -> str:
    """Return the stage key with the largest observed max latency."""
    candidates = {
        key: value
        for key, value in maximums.items()
        if key in {
            "queue_wait_ms",
            "rpm_total_ms",
            "motion_detection_ms",
            "ftl_ingest_ms",
            "ftl_get_frame_total_ms",
            "object_detection_total_ms",
            "face_detection_total_ms",
            "face_recognition_total_ms",
            "person_directory_lookup_total_ms",
            "output_build_ms",
        }
    }
    if not candidates:
        return "unknown"
    return max(candidates, key=lambda key: candidates[key])


def _write_runtime_trace_summary_json(output_dir: Path, summary: dict[str, Any]) -> Path:
    """Write runtime trace aggregate diagnostics JSON."""
    summary_path = output_dir / RUNTIME_TRACE_SUMMARY_NAME
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary_path


def _build_runtime_trace_per_frame_payload(
    records: list[ReplayFrameRecord],
) -> dict[str, Any]:
    """Build per-frame algorithm count/call/timing diagnostics payload."""
    frames_payload = [_build_runtime_trace_per_frame_entry(record) for record in records]
    warmup_frames = sum(1 for record in records if _is_warmup_record(record))
    measured_frames = len(records) - warmup_frames
    total_events = sum(
        int(frame["totals_for_frame"]["total_events_all_algorithms"])
        for frame in frames_payload
    )
    total_calls = sum(
        int(frame["totals_for_frame"]["total_algorithm_calls_all_algorithms"])
        for frame in frames_payload
    )
    total_time_ms = sum(
        float(frame["totals_for_frame"]["total_time_all_algorithms_ms"])
        for frame in frames_payload
    )
    return {
        "frames_total": len(records),
        "warmup_frames_total": warmup_frames,
        "measured_frames_total": measured_frames,
        "frames": frames_payload,
        "run_totals": {
            "events_all_frames": total_events,
            "algorithm_calls_all_frames": total_calls,
            "total_time_all_algorithms_all_frames_ms": round(total_time_ms, 3),
        },
    }


def _build_runtime_trace_per_frame_entry(record: ReplayFrameRecord) -> dict[str, Any]:
    """Build the per-frame JSON payload for one replay frame."""
    motion_calls = 1 if record.motion_stage_ran else 0
    motion_events = max(record.final_motion_regions_count, record.motion_regions_count)
    object_events = max(record.persons_count, 0)
    face_detection_events = max(record.face_rois_count, 0)
    face_recognition_events = max(record.recognized_faces_count, 0)
    motion_ms = float(record.motion_detection_ms or 0.0)
    object_ms = float(record.object_detection_total_ms or 0.0)
    face_detection_ms = float(record.face_detection_total_ms or 0.0)
    face_recognition_ms = float(record.face_recognition_total_ms or 0.0)
    total_events = (
        motion_events
        + object_events
        + face_detection_events
        + face_recognition_events
    )
    total_calls = (
        motion_calls
        + record.object_detection_call_count
        + record.face_detection_call_count
        + record.face_recognition_call_count
    )
    total_time_ms = motion_ms + object_ms + face_detection_ms + face_recognition_ms
    return {
        "frame_id": record.frame_id,
        "frame_index": record.frame_index,
        "timestamp_ms": record.timestamp_ms,
        "is_warmup": _is_warmup_record(record),
        "queue_to_aps": {
            "queue_received_at_ms": record.submitted_at_ms,
            "aps_enqueue_at_ms": record.ips_enqueued_at_ms,
            "queue_to_aps_latency_ms": round(
                float(_delta_ms(record.submitted_at_ms, record.ips_enqueued_at_ms) or 0.0),
                3,
            ),
        },
        "motion_detection": {
            "moving_zones_detected_count": motion_events,
            "algorithm_call_count": motion_calls,
            "total_events": motion_events,
            "total_time_ms": round(motion_ms, 3),
        },
        "object_detection": {
            "detected_rectangles_count": object_events,
            "algorithm_call_count": record.object_detection_call_count,
            "total_events": object_events,
            "total_time_ms": round(object_ms, 3),
        },
        "face_detection": {
            "detected_face_zones_count": face_detection_events,
            "algorithm_call_count": record.face_detection_call_count,
            "total_events": face_detection_events,
            "total_time_ms": round(face_detection_ms, 3),
        },
        "face_recognition": {
            "recognized_faces_count": face_recognition_events,
            "algorithm_call_count": record.face_recognition_call_count,
            "total_events": face_recognition_events,
            "total_time_ms": round(face_recognition_ms, 3),
        },
        "totals_for_frame": {
            "total_events_all_algorithms": total_events,
            "total_algorithm_calls_all_algorithms": total_calls,
            "total_time_all_algorithms_ms": round(total_time_ms, 3),
        },
    }


def _write_runtime_trace_per_frame_json(
    output_dir: Path,
    records: list[ReplayFrameRecord],
) -> Path:
    """Write frame-level algorithm diagnostics JSON."""
    per_frame_path = output_dir / RUNTIME_TRACE_PER_FRAME_NAME
    payload = _build_runtime_trace_per_frame_payload(records)
    with open(per_frame_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return per_frame_path


def _build_summary(
    input_video_path: Path,
    output_dir: Path,
    overlay_path: Path,
    side_by_side_path: Path | None,
    metrics_path: Path,
    sample_frames_dir: Path | None,
    records: list[ReplayFrameRecord],
    final_health: Any,
    gateway_health: Any,
    service: ImageProcessingService,
    components: Any,
    warmup_records: list[ReplayFrameRecord],
    trace_summary: dict[str, Any],
    average_fps: float,
) -> dict[str, Any]:
    """Build the summary payload written to summary.json."""
    latencies = [
        record.processing_latency_ms
        for record in records
        if record.processing_latency_ms is not None
    ]
    frames_processed = sum(1 for record in records if record.output is not None)
    frames_dropped = int(final_health.frames_dropped_oldest_total) + int(
        final_health.frames_dropped_newest_total
    ) + int(gateway_health.sink_enqueue_rejected_total)
    total_person_detections = sum(record.persons_count for record in records)
    total_face_detections = sum(record.faces_count for record in records)
    total_motion_regions_detected = sum(
        max(len(record.motion_roi_bboxes), record.motion_regions_count) for record in records
    )
    total_demo_bypass_frames = sum(1 for record in records if record.demo_bypass_motion_gate)
    total_recognized_faces = sum(record.recognized_faces_count for record in records)
    return {
        "input_video_path": str(input_video_path),
        "output_dir": str(output_dir),
        "output_overlay_path": str(overlay_path),
        "output_side_by_side_path": None if side_by_side_path is None else str(side_by_side_path),
        "metrics_csv_path": str(metrics_path),
        "runtime_trace_csv_path": str(output_dir / RUNTIME_TRACE_NAME),
        "runtime_trace_summary_path": str(output_dir / RUNTIME_TRACE_SUMMARY_NAME),
        "runtime_trace_per_frame_path": str(output_dir / RUNTIME_TRACE_PER_FRAME_NAME),
        "sample_frames_dir": None if sample_frames_dir is None else str(sample_frames_dir),
        "frames_read": len(records),
        "frames_submitted": len(records),
        "frames_processed": frames_processed,
        "warmup_frames_submitted": len(warmup_records),
        "warmup_frames_processed": sum(1 for record in warmup_records if record.output is not None),
        "frames_rendered": len(records),
        "frames_dropped": frames_dropped,
        "stale_frames_dropped": int(final_health.stale_frames_dropped_total),
        "average_fps": float(average_fps),
        "average_processing_latency_ms": _average(latencies),
        "max_processing_latency_ms": _maximum(latencies),
        "total_motion_regions_detected": total_motion_regions_detected,
        "demo_bypass_motion_gate_used": total_demo_bypass_frames > 0,
        "total_demo_bypass_frames": total_demo_bypass_frames,
        "total_object_detections": total_person_detections,
        "total_object_detection_stage_ran": int(
            trace_summary.get("total_object_detection_stage_ran", 0)
        ),
        "total_face_detection_stage_ran": int(
            trace_summary.get("total_face_detection_stage_ran", 0)
        ),
        "total_person_detections": total_person_detections,
        "total_face_detections": total_face_detections,
        "total_recognized_faces": total_recognized_faces,
        "gateway_frames_in_total": int(gateway_health.frames_in_total),
        "gateway_frames_published_total": int(gateway_health.frames_published_total),
        "ips_accepted_total": int(service._metrics.get_camera_counter(CAMERA_ID, "accepted_per_camera")),
        "ips_processed_total": int(service._metrics.get_camera_counter(CAMERA_ID, "processed_per_camera")),
        "final_service_state": str(final_health.state),
        "final_active_workers": int(final_health.active_processing_workers),
        "ftl_calls_last_frame": int(
            components.rpm.get_last_frame_metrics().get("total_ftl_calls_per_frame", 0)
        ),
        "warmup": trace_summary.get("warmup_section", {}),
        "backend_diagnostics": trace_summary.get("backend_diagnostics", {}),
        "render_mode": {
            "side_by_side_enabled": side_by_side_path is not None,
        },
    }


def _average(values: list[float]) -> float:
    """Return the average of values or 0.0 when empty."""
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _maximum(values: list[float]) -> float:
    """Return the maximum of values or 0.0 when empty."""
    if not values:
        return 0.0
    return float(max(values))


def _write_summary_json(output_dir: Path, summary: dict[str, Any]) -> Path:
    """Write the replay summary payload to JSON."""
    summary_path = output_dir / SUMMARY_NAME
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary_path


def _result_from_summary(summary: dict[str, Any]) -> VisualReplayRunResult:
    """Convert the summary payload into the public run result dataclass."""
    side_by_side_raw = summary.get("output_side_by_side_path")
    sample_frames_raw = summary.get("sample_frames_dir")
    return VisualReplayRunResult(
        output_overlay_path=Path(str(summary["output_overlay_path"])),
        summary_path=Path(str(summary["summary_path"])),
        metrics_csv_path=Path(str(summary["metrics_csv_path"])),
        output_side_by_side_path=None if side_by_side_raw is None else Path(str(side_by_side_raw)),
        sample_frames_dir=None if sample_frames_raw is None else Path(str(sample_frames_raw)),
        frames_read=int(summary["frames_read"]),
        frames_submitted=int(summary["frames_submitted"]),
        frames_processed=int(summary["frames_processed"]),
        warmup_frames_submitted=int(summary.get("warmup_frames_submitted", 0)),
        warmup_frames_processed=int(summary.get("warmup_frames_processed", 0)),
        frames_rendered=int(summary["frames_rendered"]),
        frames_dropped=int(summary["frames_dropped"]),
        stale_frames_dropped=int(summary["stale_frames_dropped"]),
        average_fps=float(summary.get("average_fps", 0.0)),
        average_processing_latency_ms=float(summary["average_processing_latency_ms"]),
        max_processing_latency_ms=float(summary["max_processing_latency_ms"]),
        total_person_detections=int(summary["total_person_detections"]),
        total_face_detections=int(summary["total_face_detections"]),
        total_motion_regions_detected=int(summary.get("total_motion_regions_detected", 0)),
        total_recognized_faces=int(summary.get("total_recognized_faces", 0)),
        gateway_frames_in_total=int(summary["gateway_frames_in_total"]),
        gateway_frames_published_total=int(summary["gateway_frames_published_total"]),
        ips_processed_total=int(summary["ips_processed_total"]),
        ftl_calls_last_frame=int(summary["ftl_calls_last_frame"]),
        final_service_state=str(summary["final_service_state"]),
        final_active_workers=int(summary["final_active_workers"]),
    )


def run_visual_replay(
    input_video_path: Path | None = None,
    output_dir: Path | None = None,
    replay_fps: float | None = None,
    max_frames: int | None = None,
    warmup_frames: int = DEFAULT_WARMUP_FRAMES,
    enable_side_by_side: bool = False,
    sample_frame_interval: int = 0,
    quality_mode: bool = False,
    debug_trace_hud: bool = False,
    rpm_config_path: Path = DEFAULT_RPM_CONFIG_PATH,
    ips_config_path: Path = DEFAULT_IPS_CONFIG_PATH,
    demo_mode: bool = False,
    summary_screen_seconds: float = 0.0,
    demo_bypass_motion_gate: bool = False,
) -> VisualReplayRunResult:
    """Run the real single-camera visual replay and return output metadata."""
    if demo_mode and summary_screen_seconds <= 0.0:
        summary_screen_seconds = DEMO_SUMMARY_SCREEN_SECONDS
    target_video_path = _resolve_input_video_path(input_video_path)
    target_output_dir, replay_fps = _resolve_replay_mode_defaults(
        quality_mode,
        output_dir,
        replay_fps,
    )
    _clean_output_directory(target_output_dir)
    frames, effective_fps = _read_video_frames(target_video_path, replay_fps, max_frames)
    if warmup_frames < 0:
        raise VisualReplayError("warmup_frames must be >= 0")
    if warmup_frames >= len(frames) and frames:
        raise VisualReplayError(
            "warmup_frames must be smaller than the number of replay frames so measured replay remains."
        )
    warmup_count = min(warmup_frames, len(frames))
    warmup_input_frames = frames[:warmup_count]
    measured_input_frames = frames[warmup_count:]
    result_handler = RecordingResultHandler()
    trace_collector = RuntimeTraceCollector()
    service, transport, components = _build_service_for_frames(
        frames,
        result_handler,
        quality_mode,
        rpm_config_path,
        ips_config_path,
        demo_bypass_motion_gate,
    )
    result_handler.attach_rpm(components.rpm)
    trace_collector.install(components.rpm)
    trace_collector.install_process_frame_hook(components.rpm)
    trace_collector.install_sink_hook(service._runtime.sink)
    service.start()
    warmup_records: list[ReplayFrameRecord] = []
    base_timestamp_ms = int(time.time() * 1000)
    try:
        warmup_records = _submit_frames(
            warmup_input_frames,
            effective_fps,
            service,
            transport,
            trace_collector,
            frame_index_offset=0,
            frame_id_prefix="warmup",
            is_warmup=True,
            base_timestamp_ms=base_timestamp_ms,
        )
        if warmup_records:
            _wait_for_phase_completion(service, len(warmup_records))
        records = _submit_frames(
            measured_input_frames,
            effective_fps,
            service,
            transport,
            trace_collector,
            frame_index_offset=len(warmup_records),
            frame_id_prefix="frame",
            is_warmup=False,
            base_timestamp_ms=base_timestamp_ms,
        )
        _stop_service(service)
    finally:
        if service.health().state not in {"STOPPED", "DEGRADED"}:
            _stop_service(service)
    final_health = service.health()
    gateway_health = service._runtime.gateway.health()
    frames_dropped = int(final_health.frames_dropped_oldest_total) + int(
        final_health.frames_dropped_newest_total
    ) + int(gateway_health.sink_enqueue_rejected_total)
    all_records = warmup_records + records
    _merge_results(
        all_records,
        result_handler,
        trace_collector,
        gateway_health,
        final_health,
        service,
        int(final_health.stale_frames_dropped_total),
        frames_dropped,
    )
    render_start = time.monotonic()
    overlay_path, side_by_side_path, sample_frames_dir, summary_frame_count = _render_videos(
        records,
        target_output_dir,
        effective_fps,
        enable_side_by_side,
        sample_frame_interval,
        debug_trace_hud,
        demo_mode=demo_mode,
        summary_screen_seconds=summary_screen_seconds,
        overlay_video_name=DEMO_OVERLAY_VIDEO_NAME if demo_mode else OVERLAY_VIDEO_NAME,
    )
    render_seconds = max(time.monotonic() - render_start, 1e-6)
    average_fps = len(records) / render_seconds
    metrics_path = _write_metrics_csv(target_output_dir, records)
    _write_runtime_trace_csv(target_output_dir, all_records)
    trace_summary = _build_runtime_trace_summary(all_records)
    _write_runtime_trace_summary_json(target_output_dir, trace_summary)
    _write_runtime_trace_per_frame_json(target_output_dir, all_records)
    summary = _build_summary(
        target_video_path,
        target_output_dir,
        overlay_path,
        side_by_side_path,
        metrics_path,
        sample_frames_dir,
        records,
        final_health,
        gateway_health,
        service,
        components,
        warmup_records,
        trace_summary,
        average_fps,
    )
    summary["demo_mode"] = demo_mode
    summary["summary_screen_frames"] = summary_frame_count
    summary_path = _write_summary_json(target_output_dir, summary)
    summary["summary_path"] = str(summary_path)
    _print_console_report(summary, trace_summary)
    return _result_from_summary(summary)


def _print_console_report(summary: dict[str, Any], trace_summary: dict[str, Any]) -> None:
    """Print a concise runtime backend and warmup report for CLI runs."""
    backend_diagnostics = trace_summary.get("backend_diagnostics", {})
    warmup = trace_summary.get("warmup_section", {})
    averages = trace_summary.get("average_latency_ms_by_stage", {})
    maximums = trace_summary.get("max_latency_ms_by_stage", {})
    print("Demo video statistics:")
    print(f"  Total frames processed: {summary.get('frames_processed', 0)}")
    print(f"  Average FPS: {float(summary.get('average_fps', 0.0)):.2f}")
    print(f"  Total motion regions detected: {summary.get('total_motion_regions_detected', 0)}")
    print(
        "  Demo bypass motion gate used: "
        f"{bool(summary.get('demo_bypass_motion_gate_used', False))}"
    )
    print(
        "  demo_bypass_motion_gate="
        f"{'true' if summary.get('demo_bypass_motion_gate_used', False) else 'false'}"
    )
    print(f"  Demo bypass frames: {summary.get('total_demo_bypass_frames', 0)}")
    print(
        "  Total object detection stage ran: "
        f"{summary.get('total_object_detection_stage_ran', 0)}"
    )
    print(f"  Total object detections: {summary.get('total_object_detections', 0)}")
    print(
        "  Total face detection stage ran: "
        f"{summary.get('total_face_detection_stage_ran', 0)}"
    )
    print(f"  Total persons detected: {summary.get('total_person_detections', 0)}")
    print(f"  Total faces detected: {summary.get('total_face_detections', 0)}")
    print(f"  Total recognized faces: {summary.get('total_recognized_faces', 0)}")
    print(f"  Output overlay path: {summary.get('output_overlay_path', '')}")
    print("Replay diagnostics:")
    print(
        "  Warmup frames: "
        f"submitted={summary.get('warmup_frames_submitted', 0)} "
        f"processed={summary.get('warmup_frames_processed', 0)}"
    )
    print(
        "  First measured rpm_total_ms: "
        f"{float(trace_summary.get('first_measured_frame_rpm_total_ms', 0.0)):.3f}"
    )
    print(
        "  Queue wait avg/max ms: "
        f"{float(averages.get('queue_wait_ms', 0.0)):.3f} / "
        f"{float(maximums.get('queue_wait_ms', 0.0)):.3f}"
    )
    print(
        "  Warmup first/max rpm_total_ms: "
        f"{float(warmup.get('warmup_first_frame_rpm_total_ms', 0.0)):.3f} / "
        f"{float(warmup.get('warmup_max_rpm_total_ms', 0.0)):.3f}"
    )
    for component_name, component_info in backend_diagnostics.items():
        backend = component_info.get("backend", "unknown")
        device_provider = component_info.get("device_provider", "unknown")
        model_name = component_info.get(
            "model_name",
            component_info.get("embedding_model", "unknown"),
        )
        print(
            f"  {component_name}: backend={backend} device/provider={device_provider} model={model_name}"
        )


def main() -> int:
    """Run the visual replay runner from the command line."""
    args = _parse_args()
    try:
        result = run_visual_replay(
            input_video_path=args.input_video,
            output_dir=args.output_dir,
            replay_fps=args.replay_fps,
            max_frames=args.max_frames,
            warmup_frames=args.warmup_frames,
            enable_side_by_side=args.side_by_side,
            sample_frame_interval=args.sample_frame_interval,
            quality_mode=args.quality_mode,
            debug_trace_hud=args.debug_trace_hud,
            rpm_config_path=args.rpm_config_path,
            ips_config_path=args.ips_config_path,
            demo_mode=args.demo_render,
            summary_screen_seconds=args.demo_summary_seconds if args.demo_render else 0.0,
            demo_bypass_motion_gate=args.demo_bypass_motion_gate,
        )
    except VisualReplayError as exc:
        print(f"[FATAL] {exc}")
        return 1
    print(f"Overlay video: {result.output_overlay_path}")
    print(f"Summary JSON: {result.summary_path}")
    print(f"Metrics CSV: {result.metrics_csv_path}")
    if result.output_side_by_side_path is not None:
        print(f"Side-by-side video: {result.output_side_by_side_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
