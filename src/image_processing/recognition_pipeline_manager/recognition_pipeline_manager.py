"""RecognitionPipelineManager — public orchestration entry point.

Spec: RecognitionPipelineManager §8.1, §4

Public API:
    RecognitionPipelineManager.process_frame(frame_packet) -> RecognitionPipelineOutput

All dependencies are injected at construction time.  RPM holds no
per-camera mutable state — temporal frame state is owned exclusively
by the FTL implementation.
"""

from __future__ import annotations

import time
from typing import Any

from image_processing.shared.contracts import FramePacket

from .input_validator import RecognitionPipelineInputValidator
from .interfaces import (
    FaceDetectionInterface,
    FaceRecognitionInterface,
    FrameTransformationLayerInterface,
    MotionDetectionInterface,
    ObjectDetectionInterface,
    PersonDirectoryInterface,
)
from .output_builder import RecognitionPipelineOutputBuilder
from .pipeline_orchestrator import PipelineOrchestrator
from .spatial_coordinator import SpatialCoordinator
from .types import RecognitionPipelineOutput


class RecognitionPipelineManager:
    """Orchestrates the recognition pipeline for one FramePacket per invocation.

    Validates input, delegates orchestration to PipelineOrchestrator, and
    assembles the final RecognitionPipelineOutput.  Never runs pipeline stages
    directly, manages threads, stores image data, or tracks frame references.

    Dependencies are injected at construction time; all stage input contracts
    are cached at PipelineOrchestrator init time (C2).

    Args:
        ftl:         FrameTransformationLayerInterface implementation
        motion:      MotionDetectionInterface implementation
        object_det:  ObjectDetectionInterface implementation
        face_det:    FaceDetectionInterface implementation
        face_rec:    FaceRecognitionInterface implementation
        person_dir:  PersonDirectoryInterface implementation
    """

    def __init__(
        self,
        ftl: FrameTransformationLayerInterface,
        motion: MotionDetectionInterface,
        object_det: ObjectDetectionInterface,
        face_det: FaceDetectionInterface,
        face_rec: FaceRecognitionInterface,
        person_dir: PersonDirectoryInterface,
        max_motion_rois_per_frame: int = 8,
        max_person_rois_per_frame: int = 16,
        max_face_rois_per_frame: int = 32,
    ) -> None:
        spatial = SpatialCoordinator()
        self._validator = RecognitionPipelineInputValidator()
        self._orchestrator = PipelineOrchestrator(
            ftl=ftl,
            motion=motion,
            object_det=object_det,
            face_det=face_det,
            face_rec=face_rec,
            person_dir=person_dir,
            spatial_coordinator=spatial,
            max_motion_rois_per_frame=max_motion_rois_per_frame,
            max_person_rois_per_frame=max_person_rois_per_frame,
            max_face_rois_per_frame=max_face_rois_per_frame,
        )
        self._builder = RecognitionPipelineOutputBuilder()
        self._last_output_build_ms: float = 0.0

    def get_last_frame_metrics(self) -> dict[str, Any]:
        return self._orchestrator.get_last_frame_metrics()

    def get_last_output_build_ms(self) -> float:
        """Return the wall-clock ms spent in the output builder for the last frame."""
        return self._last_output_build_ms

    def process_frame(self, frame_packet: FramePacket) -> RecognitionPipelineOutput:
        """Process one FramePacket through the recognition pipeline.

        Always returns a structurally valid RecognitionPipelineOutput.
        Returns an output with ``persons = []`` on validation failure,
        cold start, ingest failure, or motion-stage failure.

        Args:
            frame_packet: The canonical frame container supplied by the caller.

        Returns:
            RecognitionPipelineOutput with full-frame bboxes and recognized
            identities.  No ROI-local coordinates, scores, or embeddings.
        """
        # Validate before any pipeline work
        try:
            self._validator.validate(frame_packet)
        except ValueError:
            return RecognitionPipelineOutput(
                frame_id="",
                camera_id="",
                timestamp_ms=0,
                persons=[],
            )

        pipeline_result = self._orchestrator.execute(frame_packet)

        _t0 = time.perf_counter()
        output = self._builder.build(
            frame_id=frame_packet.frame_id,
            camera_id=frame_packet.camera_id,
            timestamp_ms=frame_packet.timestamp_ms,
            pipeline_result=pipeline_result,
        )
        self._last_output_build_ms = (time.perf_counter() - _t0) * 1000.0
        return output
