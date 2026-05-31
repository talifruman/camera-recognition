"""PipelineOrchestrator — controls the four-stage recognition pipeline for one frame.

Spec: RecognitionPipelineManager §8.3, §8.5, §8.9, §11

Processing order (spec §8.9):
    ingest → get CURRENT/PREVIOUS for Motion → motion detection
    → for each motion region: get OD ROI → object detection → project persons
    → for each person: get FD ROI → face detection → project faces
                       → for each face: re-express landmarks → get FR ROI
                                        → recognize → PersonDirectory lookup
    → return PipelineResult

Error handling (spec §11):
    - ingest failure            → return empty PipelineResult immediately
    - PreviousFrameNotAvailable → cold start; return empty PipelineResult
    - motion get_frame failure  → return empty PipelineResult
    - OD region get_frame fail  → skip that motion region; continue
    - FD person get_frame fail  → skip that person; continue
    - FR face get_frame fail    → skip that face; continue
    - projection returns None   → skip that person/face; continue
    - FR person_found=False     → omit face entirely
    - PersonDirectory not found → append with person_name="UNKNOWN"

Image.data contract (C1):
    PipelineOrchestrator treats all Image.data buffers returned by FTL
    as read-only.  No mutation is performed on any ProcessedFrame.image.data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from image_processing.face_detection.module import FaceDetectionInput
from image_processing.face_recognition.module import FaceRecognitionInput
from image_processing.frame_transformation_layer.contracts import (
    FrameTemporalSelector,
    PreviousFrameNotAvailableError,
)
from image_processing.motion_detection.module import MotionDetectionInput, MotionInputFrame
from image_processing.object_detection.module import ObjectDetectionInput
from image_processing.shared.contracts import BoundingBox, FaceLandmarks, FramePacket, Point

from .interfaces import (
    FaceDetectionInterface,
    FaceRecognitionInterface,
    FrameTransformationLayerInterface,
    MotionDetectionInterface,
    ObjectDetectionInterface,
    PersonDirectoryInterface,
)
from .spatial_coordinator import SpatialCoordinator
from .types import PipelineResult, PersonResult, RecognizedFaceResult


class PipelineOrchestrator:
    """Executes the four-stage recognition pipeline for one FramePacket.

    Dependencies are injected at construction time.  Stage input contracts
    are queried via ``get_input_contract()`` once at init and cached
    immutably for the lifetime of this instance (C2).

    Holds no per-camera mutable state — per-camera temporal state is
    owned exclusively by the FTL implementation.
    """

    def __init__(
        self,
        ftl: FrameTransformationLayerInterface,
        motion: MotionDetectionInterface,
        object_det: ObjectDetectionInterface,
        face_det: FaceDetectionInterface,
        face_rec: FaceRecognitionInterface,
        person_dir: PersonDirectoryInterface,
        spatial_coordinator: SpatialCoordinator,
        max_motion_rois_per_frame: int = 8,
        max_person_rois_per_frame: int = 16,
        max_face_rois_per_frame: int = 32,
    ) -> None:
        self._ftl = ftl
        self._motion = motion
        self._object_det = object_det
        self._face_det = face_det
        self._face_rec = face_rec
        self._person_dir = person_dir
        self._spatial = spatial_coordinator
        self._max_motion_rois_per_frame = max(1, int(max_motion_rois_per_frame))
        self._max_person_rois_per_frame = max(1, int(max_person_rois_per_frame))
        self._max_face_rois_per_frame = max(1, int(max_face_rois_per_frame))
        self._last_frame_metrics: dict[str, Any] = self._new_frame_metrics()

        # Cache stage input contracts at init time — immutable for lifetime (C2)
        self._motion_contract = motion.get_input_contract()
        self._od_contract = object_det.get_input_contract()
        self._fd_contract = face_det.get_input_contract()
        self._fr_contract = face_rec.get_input_contract()
        self._stage_static_debug = self._build_stage_static_debug()

    @dataclass(slots=True)
    class _ExecutionState:
        frame_metrics: dict[str, Any]
        remaining_person_rois: int
        remaining_face_rois: int

        # per-frame stage flags for gate-ordering assertions
        motion_stage_ran: bool = False
        od_stage_ran: bool = False
        fd_stage_ran_once: bool = False
    def get_last_frame_metrics(self) -> dict[str, Any]:
        m = self._last_frame_metrics
        return {
            # existing counter fields (backward compat)
            "total_ftl_calls_per_frame": int(m["total_ftl_calls_per_frame"]),
            "ftl_calls_by_stage": dict(m["ftl_calls_by_stage"]),
            "motion_bboxes_raw_count": int(m["motion_bboxes_raw_count"]),
            "motion_bboxes_after_filter_count": int(m["motion_bboxes_after_filter_count"]),
            "motion_bboxes_after_merge_count": int(m["motion_bboxes_after_merge_count"]),
            "od_roi_requests_count": int(m["od_roi_requests_count"]),
            "fd_roi_requests_count": int(m["fd_roi_requests_count"]),
            "fr_roi_requests_count": int(m["fr_roi_requests_count"]),
            # new timing fields
            "ftl_ingest_ms": float(m["ftl_ingest_ms"]),
            "ftl_get_frame_total_ms": float(m["ftl_get_frame_total_ms"]),
            "ftl_get_frame_call_count": int(m["ftl_get_frame_call_count"]),
            "motion_detection_ms": float(m["motion_detection_ms"]),
            "object_detection_total_ms": float(m["object_detection_total_ms"]),
            "object_detection_call_count": int(m["object_detection_call_count"]),
            "object_detection_avg_call_ms": _safe_divide(
                float(m["object_detection_total_ms"]),
                int(m["object_detection_call_count"]),
            ),
            "object_detection_max_call_ms": float(m["object_detection_max_call_ms"]),
            "object_detection_roi_width": _safe_divide(
                int(m["object_detection_roi_width_sum"]),
                int(m["object_detection_call_count"]),
            ),
            "object_detection_roi_height": _safe_divide(
                int(m["object_detection_roi_height_sum"]),
                int(m["object_detection_call_count"]),
            ),
            "object_detection_roi_area": _safe_divide(
                int(m["object_detection_roi_area_sum"]),
                int(m["object_detection_call_count"]),
            ),
            "object_detection_largest_roi_area": int(m["object_detection_roi_area_max"]),
            "object_detection_input_size": str(m["object_detection_input_size"]),
            "object_detection_model_path": str(m["object_detection_model_path"]),
            "object_detection_model_name": str(m["object_detection_model_name"]),
            "object_detection_device_provider": str(m["object_detection_device_provider"]),
            "object_detection_cuda_available": str(m["object_detection_cuda_available"]),
            "object_detection_inference_device": str(m["object_detection_inference_device"]),
            "object_detection_why_unknown": str(m["object_detection_why_unknown"]),
            "object_detection_confidence_threshold": str(
                m["object_detection_confidence_threshold"]
            ),
            "object_detection_nms_threshold": str(m["object_detection_nms_threshold"]),
            "face_detection_backend": str(m["face_detection_backend"]),
            "face_detection_device_provider": str(m["face_detection_device_provider"]),
            "face_detection_providers": str(m["face_detection_providers"]),
            "face_detection_model_path": str(m["face_detection_model_path"]),
            "face_detection_model_name": str(m["face_detection_model_name"]),
            "face_detection_input_size": str(m["face_detection_input_size"]),
            "face_detection_confidence_threshold": str(m["face_detection_confidence_threshold"]),
            "persons_returned_per_call": list(m["persons_returned_per_call"]),
            "face_detection_total_ms": float(m["face_detection_total_ms"]),
            "face_detection_call_count": int(m["face_detection_call_count"]),
            "face_detection_avg_call_ms": _safe_divide(
                float(m["face_detection_total_ms"]),
                int(m["face_detection_call_count"]),
            ),
            "face_detection_max_call_ms": float(m["face_detection_max_call_ms"]),
            "face_detection_roi_width": _safe_divide(
                int(m["face_detection_roi_width_sum"]),
                int(m["face_detection_call_count"]),
            ),
            "face_detection_roi_height": _safe_divide(
                int(m["face_detection_roi_height_sum"]),
                int(m["face_detection_call_count"]),
            ),
            "face_detection_roi_area": _safe_divide(
                int(m["face_detection_roi_area_sum"]),
                int(m["face_detection_call_count"]),
            ),
            "face_detection_largest_roi_area": int(m["face_detection_roi_area_max"]),
            "face_recognition_backend": str(m["face_recognition_backend"]),
            "face_recognition_device_provider": str(m["face_recognition_device_provider"]),
            "face_recognition_providers": str(m["face_recognition_providers"]),
            "face_recognition_model_path": str(m["face_recognition_model_path"]),
            "face_recognition_model_name": str(m["face_recognition_model_name"]),
            "face_recognition_embedding_model": str(m["face_recognition_embedding_model"]),
            "face_recognition_embedding_dimension": str(m["face_recognition_embedding_dimension"]),
            "face_recognition_threshold": str(m["face_recognition_threshold"]),
            "face_recognition_total_ms": float(m["face_recognition_total_ms"]),
            "face_recognition_call_count": int(m["face_recognition_call_count"]),
            "motion_detection_backend": str(m["motion_detection_backend"]),
            "motion_detection_device_provider": str(m["motion_detection_device_provider"]),
            "motion_detection_implementation": str(m["motion_detection_implementation"]),
            "person_directory_lookup_total_ms": float(m["person_directory_lookup_total_ms"]),
            "person_directory_lookup_count": int(m["person_directory_lookup_count"]),
            "output_build_ms": float(m["output_build_ms"]),
            # stage state / fan-out
            "motion_stage_ran": bool(m["motion_stage_ran"]),
            "motion_detected": bool(m["motion_detected"]),
            "motion_regions_raw_count": int(m["motion_regions_raw_count"]),
            "motion_regions_dropped_by_cap": int(m["motion_regions_dropped_by_cap"]),
            "final_motion_regions_count": int(m["final_motion_regions_count"]),
            "motion_regions_dropped_by_filter": int(m["motion_regions_dropped_by_filter"]),
            "motion_regions_dropped_by_merge": int(m["motion_regions_dropped_by_merge"]),
            "motion_roi_areas": list(m["motion_roi_areas"]),
            "motion_roi_largest_area": int(m["motion_roi_largest_area"]),
            "motion_roi_smallest_area": int(m["motion_roi_smallest_area"]),
            "motion_roi_total_area": int(m["motion_roi_total_area"]),
            "person_rois_raw_count": int(m["person_rois_raw_count"]),
            "person_rois_dropped_by_cap": int(m["person_rois_dropped_by_cap"]),
            "face_rois_raw_count": int(m["face_rois_raw_count"]),
            "face_rois_dropped_by_cap": int(m["face_rois_dropped_by_cap"]),
            "recognized_faces_count": int(m["recognized_faces_count"]),
            # stop reason & gate violations
            "stop_reason": str(m["stop_reason"]),
            "gate_violations": list(m["gate_violations"]),
        }

    # -----------------------------------------------------------------------
    # Public
    # -----------------------------------------------------------------------

    def execute(self, frame_packet: FramePacket) -> PipelineResult:
        """Run the full pipeline for ``frame_packet``.

        Returns an empty ``PipelineResult`` on cold start, ingest failure,
        or motion-stage failure.  Partial results are returned when only
        individual region/person/face retrievals fail.
        """
        empty: PipelineResult = PipelineResult(persons=[])

        frame_id = frame_packet.frame_id
        camera_id = frame_packet.camera_id
        timestamp_ms = frame_packet.timestamp_ms
        frame_cache: dict[tuple[Any, ...], Any] = {}
        execution_state = self._ExecutionState(
            frame_metrics=self._new_frame_metrics(),
            remaining_person_rois=self._max_person_rois_per_frame,
            remaining_face_rois=self._max_face_rois_per_frame,
        )
        execution_state.frame_metrics.update(self._stage_static_debug)

        # Full-frame bbox constructed from FramePacket — no pixel access (spec §2.1)
        full_frame_bbox: BoundingBox = BoundingBox(
            x=0, y=0,
            width=frame_packet.width,
            height=frame_packet.height,
        )

        # Step 1 — ingest (spec §8.3, §11)
        try:
            _t0 = time.perf_counter()
            self._ftl.ingest_frame(frame_packet)
            execution_state.frame_metrics["ftl_ingest_ms"] = (time.perf_counter() - _t0) * 1000.0
        except Exception:
            execution_state.frame_metrics["stop_reason"] = "ftl_ingest_error"
            self._last_frame_metrics = execution_state.frame_metrics
            return empty

        # Step 2 — get CURRENT full frame for motion
        try:
            current_processed = self._get_frame_cached(
                cache=frame_cache,
                execution_state=execution_state,
                camera_id=camera_id,
                temporal_selector=FrameTemporalSelector.CURRENT,
                region_bbox=full_frame_bbox,
                output_type=self._motion_contract["output_image_type"],
                geometry_spec=self._motion_contract["geometry_spec"],
                stage="motion_detection",
            )
        except Exception:
            execution_state.frame_metrics["stop_reason"] = "ftl_get_frame_error"
            self._last_frame_metrics = execution_state.frame_metrics
            return empty

        # Step 3 — get PREVIOUS full frame for motion (cold-start check)
        try:
            previous_processed = self._get_frame_cached(
                cache=frame_cache,
                execution_state=execution_state,
                camera_id=camera_id,
                temporal_selector=FrameTemporalSelector.PREVIOUS,
                region_bbox=full_frame_bbox,
                output_type=self._motion_contract["output_image_type"],
                geometry_spec=self._motion_contract["geometry_spec"],
                stage="motion_detection",
            )
        except PreviousFrameNotAvailableError:
            execution_state.frame_metrics["stop_reason"] = "cold_start"
            self._last_frame_metrics = execution_state.frame_metrics
            return empty  # cold start — normal operation
        except Exception:
            execution_state.frame_metrics["stop_reason"] = "ftl_get_frame_error"
            self._last_frame_metrics = execution_state.frame_metrics
            return empty

        # Step 4 — construct MotionDetectionInput and run motion detection
        # Image.data is read-only (C1) — we pass the Image reference as-is.
        motion_input = MotionDetectionInput(
            current_frame=MotionInputFrame(
                frame_id=frame_id,
                camera_id=camera_id,
                timestamp_ms=current_processed.timestamp_ms,
                image=current_processed.image,  # read-only reference
            ),
            previous_frame=MotionInputFrame(
                frame_id=previous_processed.frame_id,
                camera_id=camera_id,
                timestamp_ms=previous_processed.timestamp_ms,
                image=previous_processed.image,  # read-only reference
            ),
        )

        try:
            _t0 = time.perf_counter()
            motion_result = self._motion.detect(motion_input)
            execution_state.frame_metrics["motion_detection_ms"] = (time.perf_counter() - _t0) * 1000.0
            execution_state.frame_metrics["motion_stage_ran"] = True
            execution_state.motion_stage_ran = True
        except Exception:
            execution_state.frame_metrics["stop_reason"] = "stage_error"
            self._last_frame_metrics = execution_state.frame_metrics
            return empty

        motion_debug_info = self._get_motion_debug_info()
        execution_state.frame_metrics["motion_bboxes_raw_count"] = int(
            motion_debug_info.get("motion_bboxes_raw_count", len(motion_result["bboxes"]))
        )
        execution_state.frame_metrics["motion_bboxes_after_filter_count"] = int(
            motion_debug_info.get("motion_bboxes_after_filter_count", len(motion_result["bboxes"]))
        )
        execution_state.frame_metrics["motion_bboxes_after_merge_count"] = int(
            motion_debug_info.get("motion_bboxes_after_merge_count", len(motion_result["bboxes"]))
        )
        execution_state.frame_metrics["motion_detected"] = bool(motion_result["detected"])
        execution_state.frame_metrics["motion_regions_raw_count"] = len(motion_result["bboxes"])

        if not motion_result["detected"]:
            execution_state.frame_metrics["stop_reason"] = "no_motion"
            self._last_frame_metrics = execution_state.frame_metrics
            return empty

        selected_motion_bboxes = self._limit_bboxes_by_area(
            motion_result["bboxes"],
            self._max_motion_rois_per_frame,
        )
        execution_state.frame_metrics["motion_regions_dropped_by_cap"] = (
            len(motion_result["bboxes"]) - len(selected_motion_bboxes)
        )
        execution_state.frame_metrics["final_motion_regions_count"] = len(selected_motion_bboxes)
        _raw = execution_state.frame_metrics["motion_bboxes_raw_count"]
        _after_filter = execution_state.frame_metrics["motion_bboxes_after_filter_count"]
        _after_merge = execution_state.frame_metrics["motion_bboxes_after_merge_count"]
        execution_state.frame_metrics["motion_regions_dropped_by_filter"] = max(0, _raw - _after_filter)
        execution_state.frame_metrics["motion_regions_dropped_by_merge"] = max(0, _after_filter - _after_merge)
        if selected_motion_bboxes:
            _roi_areas = [int(b["width"]) * int(b["height"]) for b in selected_motion_bboxes]
            execution_state.frame_metrics["motion_roi_areas"] = _roi_areas
            execution_state.frame_metrics["motion_roi_largest_area"] = max(_roi_areas)
            execution_state.frame_metrics["motion_roi_smallest_area"] = min(_roi_areas)
            execution_state.frame_metrics["motion_roi_total_area"] = sum(_roi_areas)

        if not selected_motion_bboxes:
            execution_state.frame_metrics["stop_reason"] = "no_motion_regions"
            self._last_frame_metrics = execution_state.frame_metrics
            return empty

        # Step 5 — process each motion region through OD → FD → FR
        persons: list[PersonResult] = []

        for motion_bbox in selected_motion_bboxes:
            execution_state.frame_metrics["od_roi_requests_count"] += 1
            self._process_motion_region(
                cache=frame_cache,
                execution_state=execution_state,
                camera_id=camera_id,
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                motion_bbox=motion_bbox,
                persons=persons,
            )

        execution_state.frame_metrics["stop_reason"] = _determine_stop_reason_from_state(
            execution_state.frame_metrics
        )
        self._last_frame_metrics = execution_state.frame_metrics
        return PipelineResult(persons=persons)

    # -----------------------------------------------------------------------
    # Private helpers — one per pipeline stage boundary
    # -----------------------------------------------------------------------

    def _process_motion_region(
        self,
        *,
        cache: dict[tuple[Any, ...], Any],
        execution_state: _ExecutionState,
        camera_id: str,
        frame_id: str,
        timestamp_ms: int,
        motion_bbox: BoundingBox,
        persons: list[PersonResult],
    ) -> None:
        """Object detection for one motion region + person accumulation."""
        # get OD ROI (spec §8.5.2)
        # Gate assertion: OD must only run after motion detection ran
        if not execution_state.motion_stage_ran:
            execution_state.frame_metrics["gate_violations"].append("od_without_motion")
        try:
            od_processed = self._get_frame_cached(
                cache=cache,
                execution_state=execution_state,
                camera_id=camera_id,
                temporal_selector=FrameTemporalSelector.CURRENT,
                region_bbox=motion_bbox,
                output_type=self._od_contract["output_image_type"],
                geometry_spec=self._od_contract["geometry_spec"],
                stage="object_detection",
            )
        except Exception:
            return  # skip this region (spec §11)

        od_input = ObjectDetectionInput(
            frame_id=frame_id,
            camera_id=camera_id,
            timestamp_ms=timestamp_ms,
            roi_image=od_processed.image,   # read-only (C1)
            roi_bbox_frame=motion_bbox,
        )

        try:
            roi_width = int(motion_bbox["width"])
            roi_height = int(motion_bbox["height"])
            roi_area = roi_width * roi_height
            execution_state.frame_metrics["object_detection_roi_width_sum"] += roi_width
            execution_state.frame_metrics["object_detection_roi_height_sum"] += roi_height
            execution_state.frame_metrics["object_detection_roi_area_sum"] += roi_area
            execution_state.frame_metrics["object_detection_roi_width_max"] = max(
                execution_state.frame_metrics["object_detection_roi_width_max"],
                roi_width,
            )
            execution_state.frame_metrics["object_detection_roi_height_max"] = max(
                execution_state.frame_metrics["object_detection_roi_height_max"],
                roi_height,
            )
            execution_state.frame_metrics["object_detection_roi_area_max"] = max(
                execution_state.frame_metrics["object_detection_roi_area_max"],
                roi_area,
            )
            _t0 = time.perf_counter()
            od_result = self._object_det.detect(od_input)
            _elapsed = (time.perf_counter() - _t0) * 1000.0
            execution_state.frame_metrics["object_detection_total_ms"] += _elapsed
            execution_state.frame_metrics["object_detection_call_count"] += 1
            execution_state.frame_metrics["object_detection_max_call_ms"] = max(
                execution_state.frame_metrics["object_detection_max_call_ms"],
                _elapsed,
            )
            persons_returned = len(od_result["persons"])
            execution_state.frame_metrics["persons_returned_per_call"].append(persons_returned)
            execution_state.frame_metrics["object_detection_persons_returned_total"] += persons_returned
            execution_state.od_stage_ran = True
        except Exception:
            return  # skip this region

        full_person_bboxes: list[BoundingBox] = []
        for roi_person_bbox in od_result["persons"]:
            # Project person bbox from OD ROI-local → full-frame (spec §8.5.2)
            full_person_bbox = self._spatial.project_bbox_to_full_frame(
                roi_person_bbox,
                od_processed.source_bbox_full_frame,
                od_processed.spatial_transform,
            )
            if full_person_bbox is None:
                continue  # invalid projection — skip (spec §11)

            full_person_bboxes.append(full_person_bbox)

        if execution_state.remaining_person_rois <= 0:
            return

        execution_state.frame_metrics["person_rois_raw_count"] += len(full_person_bboxes)
        allowed_persons = self._limit_bboxes_by_area(
            full_person_bboxes,
            min(execution_state.remaining_person_rois, len(full_person_bboxes)),
        )
        execution_state.frame_metrics["person_rois_dropped_by_cap"] += (
            len(full_person_bboxes) - len(allowed_persons)
        )
        execution_state.remaining_person_rois -= len(allowed_persons)

        for full_person_bbox in allowed_persons:

            person_result = PersonResult(
                person_bbox=full_person_bbox,
                recognized_faces=[],
            )
            persons.append(person_result)

            # Face detection for this person (spec §8.5.3)
            self._process_person_faces(
                cache=cache,
                execution_state=execution_state,
                camera_id=camera_id,
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                full_person_bbox=full_person_bbox,
                person_result=person_result,
            )

    def _process_person_faces(
        self,
        *,
        cache: dict[tuple[Any, ...], Any],
        execution_state: _ExecutionState,
        camera_id: str,
        frame_id: str,
        timestamp_ms: int,
        full_person_bbox: BoundingBox,
        person_result: PersonResult,
    ) -> None:
        """Face detection + recognition for one projected person region."""
        # get FD ROI (spec §8.5.3)
        execution_state.frame_metrics["fd_roi_requests_count"] += 1
        # Gate assertion: FD must only run if OD already ran
        if not execution_state.od_stage_ran:
            execution_state.frame_metrics["gate_violations"].append("fd_without_od")
        try:
            fd_processed = self._get_frame_cached(
                cache=cache,
                execution_state=execution_state,
                camera_id=camera_id,
                temporal_selector=FrameTemporalSelector.CURRENT,
                region_bbox=full_person_bbox,
                output_type=self._fd_contract["output_image_type"],
                geometry_spec=self._fd_contract["geometry_spec"],
                stage="face_detection",
            )
        except Exception:
            return  # skip this person (spec §11)

        fd_input = FaceDetectionInput(
            frame_id=frame_id,
            camera_id=camera_id,
            timestamp_ms=timestamp_ms,
            roi_image=fd_processed.image,   # read-only (C1)
        )

        try:
            roi_width = int(full_person_bbox["width"])
            roi_height = int(full_person_bbox["height"])
            roi_area = roi_width * roi_height
            execution_state.frame_metrics["face_detection_roi_width_sum"] += roi_width
            execution_state.frame_metrics["face_detection_roi_height_sum"] += roi_height
            execution_state.frame_metrics["face_detection_roi_area_sum"] += roi_area
            execution_state.frame_metrics["face_detection_roi_width_max"] = max(
                execution_state.frame_metrics["face_detection_roi_width_max"],
                roi_width,
            )
            execution_state.frame_metrics["face_detection_roi_height_max"] = max(
                execution_state.frame_metrics["face_detection_roi_height_max"],
                roi_height,
            )
            execution_state.frame_metrics["face_detection_roi_area_max"] = max(
                execution_state.frame_metrics["face_detection_roi_area_max"],
                roi_area,
            )
            _t0 = time.perf_counter()
            fd_output = self._face_det.detect_faces(fd_input)
            _elapsed = (time.perf_counter() - _t0) * 1000.0
            execution_state.frame_metrics["face_detection_total_ms"] += _elapsed
            execution_state.frame_metrics["face_detection_call_count"] += 1
            execution_state.frame_metrics["face_detection_max_call_ms"] = max(
                execution_state.frame_metrics["face_detection_max_call_ms"],
                _elapsed,
            )
            execution_state.fd_stage_ran_once = True
        except Exception:
            return  # skip this person

        projected_faces: list[tuple[BoundingBox, FaceLandmarks]] = []
        for detected_face in fd_output["detections"]:
            roi_face_bbox = detected_face["face_bbox"]

            # Project face bbox from FD ROI-local → full-frame (spec §8.5.4)
            full_face_bbox = self._spatial.project_bbox_to_full_frame(
                roi_face_bbox,
                fd_processed.source_bbox_full_frame,
                fd_processed.spatial_transform,
            )
            if full_face_bbox is None:
                continue  # invalid projection — skip

            # Re-express landmarks to face-crop-local (spec §8.5.4)
            roi_landmarks = detected_face["landmarks"]

            # Skip face if any landmark is zero-padded (§3.2) — any Point(0,0)
            # indicates a detection with fewer than 5 real landmarks.
            _landmark_fields = ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right")
            if any(
                roi_landmarks[f]["x"] == 0 and roi_landmarks[f]["y"] == 0
                for f in _landmark_fields
            ):
                continue

            face_local_landmarks = self._to_face_local_landmarks(
                roi_landmarks, roi_face_bbox
            )

            projected_faces.append((full_face_bbox, face_local_landmarks))

        if execution_state.remaining_face_rois <= 0:
            return

        execution_state.frame_metrics["face_rois_raw_count"] += len(projected_faces)
        selected_faces = self._limit_face_candidates_by_area(
            projected_faces,
            min(execution_state.remaining_face_rois, len(projected_faces)),
        )
        execution_state.frame_metrics["face_rois_dropped_by_cap"] += (
            len(projected_faces) - len(selected_faces)
        )
        execution_state.remaining_face_rois -= len(selected_faces)

        for full_face_bbox, face_local_landmarks in selected_faces:
            execution_state.frame_metrics["fr_roi_requests_count"] += 1

            # Face recognition (spec §8.5.4)
            self._process_face_recognition(
                cache=cache,
                execution_state=execution_state,
                camera_id=camera_id,
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                full_face_bbox=full_face_bbox,
                face_local_landmarks=face_local_landmarks,
                person_result=person_result,
            )

    def _process_face_recognition(
        self,
        *,
        cache: dict[tuple[Any, ...], Any],
        execution_state: _ExecutionState,
        camera_id: str,
        frame_id: str,
        timestamp_ms: int,
        full_face_bbox: BoundingBox,
        face_local_landmarks: FaceLandmarks,
        person_result: PersonResult,
    ) -> None:
        """Face recognition for one projected face region + PersonDirectory lookup."""
        # get FR ROI (spec §8.5.4)
        # Gate assertion: FR must only run if FD already ran
        if not execution_state.fd_stage_ran_once:
            execution_state.frame_metrics["gate_violations"].append("fr_without_fd")
        try:
            fr_processed = self._get_frame_cached(
                cache=cache,
                execution_state=execution_state,
                camera_id=camera_id,
                temporal_selector=FrameTemporalSelector.CURRENT,
                region_bbox=full_face_bbox,
                output_type=self._fr_contract["output_image_type"],
                geometry_spec=self._fr_contract["geometry_spec"],
                stage="face_recognition",
            )
        except Exception:
            return  # skip this face (spec §11)

        fr_input = FaceRecognitionInput(
            frame_id=frame_id,
            camera_id=camera_id,
            timestamp_ms=timestamp_ms,
            face_roi_image=fr_processed.image,   # read-only (C1)
            landmarks=face_local_landmarks,
        )

        try:
            _t0 = time.perf_counter()
            fr_output = self._face_rec.recognize(fr_input)
            _elapsed = (time.perf_counter() - _t0) * 1000.0
            execution_state.frame_metrics["face_recognition_total_ms"] += _elapsed
            execution_state.frame_metrics["face_recognition_call_count"] += 1
        except Exception:
            return  # skip this face

        if not fr_output["person_found"]:
            return  # unrecognized — omit entirely (spec §11)

        # PersonDirectory lookup for person_name enrichment (spec §8.5.4)
        recognized_person_id = fr_output["person_id"]
        try:
            _t0 = time.perf_counter()
            pd_output = self._person_dir.get_person(recognized_person_id)
            _elapsed = (time.perf_counter() - _t0) * 1000.0
            execution_state.frame_metrics["person_directory_lookup_total_ms"] += _elapsed
            execution_state.frame_metrics["person_directory_lookup_count"] += 1
        except Exception:
            pd_output = {"person_id": "UNKNOWN", "person_name": "UNKNOWN", "found": False}

        person_result["recognized_faces"].append(
            RecognizedFaceResult(
                face_bbox=full_face_bbox,
                person_id=pd_output["person_id"],
                person_name=pd_output["person_name"],
                found=pd_output["found"],
            )
        )
        execution_state.frame_metrics["recognized_faces_count"] += 1

    def _get_frame_cached(
        self,
        *,
        cache: dict[tuple[Any, ...], Any],
        execution_state: _ExecutionState,
        camera_id: str,
        temporal_selector: FrameTemporalSelector,
        region_bbox: BoundingBox,
        output_type: Any,
        geometry_spec: Any,
        stage: str,
    ) -> Any:
        """Fetch ProcessedFrame once per unique request key within a frame."""
        key = self._make_frame_cache_key(
            camera_id,
            temporal_selector,
            region_bbox,
            output_type,
            geometry_spec,
        )
        if key in cache:
            return cache[key]

        _t0 = time.perf_counter()
        processed = self._ftl.get_frame(
            camera_id,
            temporal_selector,
            region_bbox,
            output_type,
            geometry_spec,
        )
        execution_state.frame_metrics["ftl_get_frame_total_ms"] += (time.perf_counter() - _t0) * 1000.0
        execution_state.frame_metrics["ftl_get_frame_call_count"] += 1
        cache[key] = processed
        execution_state.frame_metrics["total_ftl_calls_per_frame"] += 1
        if stage in execution_state.frame_metrics["ftl_calls_by_stage"]:
            execution_state.frame_metrics["ftl_calls_by_stage"][stage] += 1
        return processed

    @staticmethod
    def _limit_bboxes_by_area(
        bboxes: list[BoundingBox],
        max_count: int,
    ) -> list[BoundingBox]:
        if max_count <= 0:
            return []
        sorted_bboxes = sorted(
            bboxes,
            key=lambda bbox: int(bbox["width"]) * int(bbox["height"]),
            reverse=True,
        )
        return sorted_bboxes[:max_count]

    @staticmethod
    def _limit_face_candidates_by_area(
        faces: list[tuple[BoundingBox, FaceLandmarks]],
        max_count: int,
    ) -> list[tuple[BoundingBox, FaceLandmarks]]:
        if max_count <= 0:
            return []
        sorted_faces = sorted(
            faces,
            key=lambda entry: int(entry[0]["width"]) * int(entry[0]["height"]),
            reverse=True,
        )
        return sorted_faces[:max_count]

    def _get_motion_debug_info(self) -> dict[str, Any]:
        get_debug_info = getattr(self._motion, "get_last_debug_info", None)
        if not callable(get_debug_info):
            return {}

        try:
            value = get_debug_info()
        except Exception:
            return {}

        return value if isinstance(value, dict) else {}

    def _build_stage_static_debug(self) -> dict[str, Any]:
        """Collect static backend diagnostics from configured stage modules."""
        od_info = self._get_component_backend_info(self._object_det)
        fd_info = self._get_component_backend_info(self._face_det)
        fr_info = self._get_component_backend_info(self._face_rec)
        motion_info = self._get_component_backend_info(self._motion)
        config_obj = self._extract_od_config_object(self._object_det)
        model_path = self._coalesce_value(
            str(od_info.get("model_path", "unknown")),
            self._extract_attr(config_obj, "model_path"),
        )
        confidence_threshold = self._coalesce_value(
            str(od_info.get("confidence_threshold", "unknown")),
            self._extract_attr(config_obj, "person_confidence_threshold"),
        )
        nms_threshold = self._coalesce_value(
            str(od_info.get("nms_threshold", "unknown")),
            self._extract_attr(config_obj, "nms_iou_threshold"),
        )
        provider = self._coalesce_value(
            str(od_info.get("device_provider", "unknown")),
            self._extract_attr(config_obj, "inference_backend"),
        )
        input_size = self._coalesce_value(
            str(od_info.get("input_size", "unknown")),
            self._format_input_size(self._od_contract),
        )
        return {
            "object_detection_model_path": model_path,
            "object_detection_model_name": self._extract_model_name(model_path),
            "object_detection_device_provider": provider,
            "object_detection_cuda_available": self._coalesce_value(
                str(od_info.get("cuda_available", "unknown")), "unknown"
            ),
            "object_detection_inference_device": self._coalesce_value(
                str(od_info.get("inference_device", "unknown")), "unknown"
            ),
            "object_detection_why_unknown": str(od_info.get("why_unknown", "")),
            "object_detection_confidence_threshold": confidence_threshold,
            "object_detection_nms_threshold": nms_threshold,
            "object_detection_input_size": input_size,
            "face_detection_backend": str(fd_info.get("backend", "unknown")),
            "face_detection_device_provider": str(fd_info.get("device_provider", "unknown")),
            "face_detection_providers": str(fd_info.get("providers", "unknown")),
            "face_detection_model_path": str(fd_info.get("model_path", "unknown")),
            "face_detection_model_name": str(fd_info.get("model_name", "unknown")),
            "face_detection_input_size": str(fd_info.get("input_size", "unknown")),
            "face_detection_confidence_threshold": str(fd_info.get("confidence_threshold", "unknown")),
            "face_recognition_backend": str(fr_info.get("backend", "unknown")),
            "face_recognition_device_provider": str(fr_info.get("device_provider", "unknown")),
            "face_recognition_providers": str(fr_info.get("providers", "unknown")),
            "face_recognition_model_path": str(fr_info.get("model_path", "unknown")),
            "face_recognition_model_name": str(fr_info.get("model_name", "unknown")),
            "face_recognition_embedding_model": str(fr_info.get("embedding_model", "unknown")),
            "face_recognition_embedding_dimension": str(fr_info.get("embedding_dimension", "unknown")),
            "face_recognition_threshold": str(fr_info.get("recognition_threshold", "unknown")),
            "motion_detection_backend": str(motion_info.get("backend", "unknown")),
            "motion_detection_device_provider": str(motion_info.get("device_provider", "unknown")),
            "motion_detection_implementation": str(motion_info.get("implementation", "unknown")),
        }

    @staticmethod
    def _get_component_backend_info(component: Any) -> dict[str, Any]:
        """Return backend diagnostics from a component when exposed."""
        getter = getattr(component, "get_backend_info", None)
        if not callable(getter):
            return {}
        try:
            info = getter()
        except Exception:
            return {}
        return info if isinstance(info, dict) else {}

    @staticmethod
    def _extract_od_config_object(object_det: Any) -> Any:
        """Return the OD config object from wrapper/delegate chain when available."""
        delegate = getattr(object_det, "_delegate", object_det)
        return getattr(delegate, "_config", None)

    @staticmethod
    def _extract_attr(value: Any, attr_name: str) -> str:
        """Return a safe string representation of an attribute value."""
        if value is None:
            return "unavailable"
        attr_value = getattr(value, attr_name, None)
        if attr_value is None:
            return "unavailable"
        return str(attr_value)

    @staticmethod
    def _coalesce_value(primary: str, fallback: str) -> str:
        """Return the first non-empty diagnostic value without guessing."""
        primary_text = str(primary).strip()
        if primary_text and primary_text not in {"unknown", "unavailable"}:
            return primary_text
        fallback_text = str(fallback).strip()
        if fallback_text:
            return fallback_text
        return "unknown"

    @staticmethod
    def _format_input_size(contract: dict[str, Any]) -> str:
        """Format OD input size from stage contract as WIDTHxHEIGHT string."""
        geometry = contract.get("geometry_spec") if isinstance(contract, dict) else None
        if not isinstance(geometry, dict):
            return "unavailable"
        width = geometry.get("width")
        height = geometry.get("height")
        if width is None or height is None:
            return "unavailable"
        return f"{int(width)}x{int(height)}"

    @staticmethod
    def _extract_model_name(model_path: str) -> str:
        """Extract model filename from a configured path-like string."""
        if not model_path or model_path == "unavailable":
            return "unavailable"
        normalized = model_path.replace("\\", "/")
        parts = normalized.split("/")
        return parts[-1] if parts else "unavailable"

    @staticmethod
    def _new_frame_metrics() -> dict[str, Any]:
        return {
            # existing counter fields (backward compat)
            "total_ftl_calls_per_frame": 0,
            "ftl_calls_by_stage": {
                "motion_detection": 0,
                "object_detection": 0,
                "face_detection": 0,
                "face_recognition": 0,
                "debug_overlay": 0,
            },
            "motion_bboxes_raw_count": 0,
            "motion_bboxes_after_filter_count": 0,
            "motion_bboxes_after_merge_count": 0,
            "od_roi_requests_count": 0,
            "fd_roi_requests_count": 0,
            "fr_roi_requests_count": 0,
            # new timing fields
            "ftl_ingest_ms": 0.0,
            "ftl_get_frame_total_ms": 0.0,
            "ftl_get_frame_call_count": 0,
            "motion_detection_ms": 0.0,
            "object_detection_total_ms": 0.0,
            "object_detection_call_count": 0,
            "object_detection_max_call_ms": 0.0,
            "object_detection_roi_width_sum": 0,
            "object_detection_roi_height_sum": 0,
            "object_detection_roi_area_sum": 0,
            "object_detection_roi_width_max": 0,
            "object_detection_roi_height_max": 0,
            "object_detection_roi_area_max": 0,
            "persons_returned_per_call": [],
            "object_detection_persons_returned_total": 0,
            "object_detection_input_size": "unavailable",
            "object_detection_model_path": "unavailable",
            "object_detection_model_name": "unavailable",
            "object_detection_device_provider": "unavailable",
            "object_detection_cuda_available": "unknown",
            "object_detection_inference_device": "unknown",
            "object_detection_why_unknown": "",
            "object_detection_confidence_threshold": "unavailable",
            "object_detection_nms_threshold": "unavailable",
            "face_detection_backend": "unknown",
            "face_detection_device_provider": "unknown",
            "face_detection_providers": "unknown",
            "face_detection_model_path": "unknown",
            "face_detection_model_name": "unknown",
            "face_detection_input_size": "unknown",
            "face_detection_confidence_threshold": "unknown",
            "face_detection_total_ms": 0.0,
            "face_detection_call_count": 0,
            "face_detection_max_call_ms": 0.0,
            "face_detection_roi_width_sum": 0,
            "face_detection_roi_height_sum": 0,
            "face_detection_roi_area_sum": 0,
            "face_detection_roi_width_max": 0,
            "face_detection_roi_height_max": 0,
            "face_detection_roi_area_max": 0,
            "face_recognition_backend": "unknown",
            "face_recognition_device_provider": "unknown",
            "face_recognition_providers": "unknown",
            "face_recognition_model_path": "unknown",
            "face_recognition_model_name": "unknown",
            "face_recognition_embedding_model": "unknown",
            "face_recognition_embedding_dimension": "unknown",
            "face_recognition_threshold": "unknown",
            "face_recognition_total_ms": 0.0,
            "face_recognition_call_count": 0,
            "motion_detection_backend": "unknown",
            "motion_detection_device_provider": "unknown",
            "motion_detection_implementation": "unknown",
            "person_directory_lookup_total_ms": 0.0,
            "person_directory_lookup_count": 0,
            "output_build_ms": 0.0,
            # stage state / fan-out
            "motion_stage_ran": False,
            "motion_detected": False,
            "motion_regions_raw_count": 0,
            "motion_regions_dropped_by_cap": 0,
            "final_motion_regions_count": 0,
            "motion_regions_dropped_by_filter": 0,
            "motion_regions_dropped_by_merge": 0,
            "motion_roi_areas": [],
            "motion_roi_largest_area": 0,
            "motion_roi_smallest_area": 0,
            "motion_roi_total_area": 0,
            "person_rois_raw_count": 0,
            "person_rois_dropped_by_cap": 0,
            "face_rois_raw_count": 0,
            "face_rois_dropped_by_cap": 0,
            "recognized_faces_count": 0,
            # stop reason & gate violations
            "stop_reason": "not_run",
            "gate_violations": [],
        }

    @staticmethod
    def _make_frame_cache_key(
        camera_id: str,
        temporal_selector: FrameTemporalSelector,
        region_bbox: BoundingBox,
        output_type: Any,
        geometry_spec: Any,
    ) -> tuple[Any, ...]:
        resize_policy = geometry_spec["resize_policy"]
        resize_value = resize_policy.value if hasattr(resize_policy, "value") else str(resize_policy)
        output_value = output_type.value if hasattr(output_type, "value") else str(output_type)
        return (
            camera_id,
            temporal_selector.value,
            int(region_bbox["x"]),
            int(region_bbox["y"]),
            int(region_bbox["width"]),
            int(region_bbox["height"]),
            output_value,
            int(geometry_spec["width"]),
            int(geometry_spec["height"]),
            resize_value,
        )

    # -----------------------------------------------------------------------
    # Static helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _to_face_local_landmarks(
        roi_landmarks: FaceLandmarks,
        face_bbox: BoundingBox,
    ) -> FaceLandmarks:
        """Re-express landmarks from person-ROI-local to face-crop-local.

        Subtracts the face bbox origin from each landmark (spec §8.5.4):
            face_local_lm.x = lm.x - face_bbox.x
            face_local_lm.y = lm.y - face_bbox.y
        """
        bx = face_bbox["x"]
        by = face_bbox["y"]

        def shift(pt: Point) -> Point:
            return Point(x=pt["x"] - bx, y=pt["y"] - by)

        return FaceLandmarks(
            left_eye=shift(roi_landmarks["left_eye"]),
            right_eye=shift(roi_landmarks["right_eye"]),
            nose=shift(roi_landmarks["nose"]),
            mouth_left=shift(roi_landmarks["mouth_left"]),
            mouth_right=shift(roi_landmarks["mouth_right"]),
        )


def _determine_stop_reason_from_state(frame_metrics: dict) -> str:
    """Determine the final stop_reason after the full pipeline loop completes."""
    if not frame_metrics["motion_stage_ran"]:
        return "cold_start"
    if not frame_metrics["motion_detected"]:
        return "no_motion"
    if frame_metrics["object_detection_call_count"] == 0:
        return "no_motion_regions"
    if frame_metrics["person_rois_raw_count"] == 0:
        return "no_persons"
    if frame_metrics["face_detection_call_count"] == 0:
        return "no_faces"
    if frame_metrics["face_recognition_call_count"] == 0:
        return "no_faces"
    if frame_metrics["recognized_faces_count"] == 0:
        return "no_recognized_faces"
    return "completed"


def _safe_divide(numerator: float | int, denominator: int) -> float:
    """Return numerator/denominator with zero-safe fallback."""
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)
