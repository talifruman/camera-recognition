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

from dataclasses import dataclass
from typing import Any

from image_processing.face_detection.module import FaceDetectionInput
from image_processing.face_recognition.module import FaceRecognitionInput
from image_processing.frame_transformation_layer.contracts import (
    FramePacket,
    FrameTemporalSelector,
    PreviousFrameNotAvailableError,
)
from image_processing.motion_detection.module import MotionDetectionInput, MotionInputFrame
from image_processing.object_detection.module import ObjectDetectionInput
from image_processing.shared.contracts import BoundingBox, FaceLandmarks, Point

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

    @dataclass(slots=True)
    class _ExecutionState:
        frame_metrics: dict[str, Any]
        remaining_person_rois: int
        remaining_face_rois: int

    def get_last_frame_metrics(self) -> dict[str, Any]:
        return {
            "total_ftl_calls_per_frame": int(self._last_frame_metrics["total_ftl_calls_per_frame"]),
            "ftl_calls_by_stage": dict(self._last_frame_metrics["ftl_calls_by_stage"]),
            "motion_bboxes_raw_count": int(self._last_frame_metrics["motion_bboxes_raw_count"]),
            "motion_bboxes_after_filter_count": int(self._last_frame_metrics["motion_bboxes_after_filter_count"]),
            "motion_bboxes_after_merge_count": int(self._last_frame_metrics["motion_bboxes_after_merge_count"]),
            "od_roi_requests_count": int(self._last_frame_metrics["od_roi_requests_count"]),
            "fd_roi_requests_count": int(self._last_frame_metrics["fd_roi_requests_count"]),
            "fr_roi_requests_count": int(self._last_frame_metrics["fr_roi_requests_count"]),
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

        # Full-frame bbox constructed from FramePacket — no pixel access (spec §2.1)
        full_frame_bbox: BoundingBox = BoundingBox(
            x=0, y=0,
            width=frame_packet.width,
            height=frame_packet.height,
        )

        # Step 1 — ingest (spec §8.3, §11)
        try:
            self._ftl.ingest_frame(frame_packet)
        except Exception:
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
            self._last_frame_metrics = execution_state.frame_metrics
            return empty  # cold start — normal operation
        except Exception:
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
            motion_result = self._motion.detect(motion_input)
        except Exception:
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

        if not motion_result["detected"]:
            self._last_frame_metrics = execution_state.frame_metrics
            return empty

        selected_motion_bboxes = self._limit_bboxes_by_area(
            motion_result["bboxes"],
            self._max_motion_rois_per_frame,
        )

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
            od_result = self._object_det.detect(od_input)
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

        allowed_persons = self._limit_bboxes_by_area(
            full_person_bboxes,
            min(execution_state.remaining_person_rois, len(full_person_bboxes)),
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
            fd_output = self._face_det.detect_faces(fd_input)
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

        selected_faces = self._limit_face_candidates_by_area(
            projected_faces,
            min(execution_state.remaining_face_rois, len(projected_faces)),
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
            fr_output = self._face_rec.recognize(fr_input)
        except Exception:
            return  # skip this face

        if not fr_output["person_found"]:
            return  # unrecognized — omit entirely (spec §11)

        # PersonDirectory lookup for person_name enrichment (spec §8.5.4)
        recognized_person_id = fr_output["person_id"]
        try:
            pd_output = self._person_dir.get_person(recognized_person_id)
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

        processed = self._ftl.get_frame(
            camera_id,
            temporal_selector,
            region_bbox,
            output_type,
            geometry_spec,
        )
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

    @staticmethod
    def _new_frame_metrics() -> dict[str, Any]:
        return {
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
