"""Protocol definitions for all RPM pipeline stage dependencies.

RPM depends exclusively on these interfaces — never on concrete
implementations.  Any compliant implementation may be substituted without
touching RPM or its tests.

Existing interfaces re-exported from their home modules:
    MotionDetectionInterface  — image_processing.motion_detection.module
    FaceRecognitionInterface  — image_processing.face_recognition.module

New interfaces defined here:
    ObjectDetectionInterface
    FaceDetectionInterface
    FrameTransformationLayerInterface
    PersonDirectoryInterface
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from image_processing.face_detection.module import (
    FaceDetectionInput,
    FaceDetectionOutput,
)
from image_processing.face_recognition.module import (
    FaceRecognitionInput,
    FaceRecognitionOutput,
    FaceRecognitionInterface,  # noqa: F401 — re-exported for callers
)
from image_processing.frame_transformation_layer.contracts import (
    FrameTemporalSelector,
    ProcessedFrame,
)
from image_processing.motion_detection.module import (
    MotionDetectionInput,
    MotionDetectionInterface,  # noqa: F401 — re-exported for callers
    MotionResult,
)
from image_processing.object_detection.module import (
    ObjectDetectionInput,
    PersonDetectionResult,
)
from image_processing.person_directory.module import PersonDirectoryOutput
from image_processing.shared.contracts import (
    BoundingBox,
    FramePacket,
    GeometrySpec,
    OutputImageType,
    PipelineStageInputContract,
)


# ---------------------------------------------------------------------------
# ObjectDetectionInterface
# ---------------------------------------------------------------------------


@runtime_checkable
class ObjectDetectionInterface(Protocol):
    """Public interface for the Object Detection stage.

    RPM depends only on this Protocol — the concrete ``ObjectDetectionModule``
    satisfies it structurally.
    """

    def detect(self, input: ObjectDetectionInput) -> PersonDetectionResult: ...

    def get_input_contract(self) -> PipelineStageInputContract: ...


# ---------------------------------------------------------------------------
# FaceDetectionInterface
# ---------------------------------------------------------------------------


@runtime_checkable
class FaceDetectionInterface(Protocol):
    """Public interface for the Face Detection stage."""

    def detect_faces(self, input: FaceDetectionInput) -> FaceDetectionOutput: ...

    def get_input_contract(self) -> PipelineStageInputContract: ...


# ---------------------------------------------------------------------------
# FrameTransformationLayerInterface
# ---------------------------------------------------------------------------


@runtime_checkable
class FrameTransformationLayerInterface(Protocol):
    """Public interface for the Frame Transformation Layer.

    RPM calls ``ingest_frame`` before any pipeline stage and ``get_frame``
    to retrieve processed frame regions.  Per-camera temporal state
    (CURRENT / PREVIOUS) is owned exclusively by the FTL implementation.
    """

    def ingest_frame(self, frame_packet: FramePacket) -> None: ...

    def get_frame(
        self,
        camera_id: str,
        temporal_selector: FrameTemporalSelector,
        region_bbox: BoundingBox,
        output_type: OutputImageType,
        geometry_spec: GeometrySpec,
    ) -> ProcessedFrame: ...


# ---------------------------------------------------------------------------
# PersonDirectoryInterface
# ---------------------------------------------------------------------------


@runtime_checkable
class PersonDirectoryInterface(Protocol):
    """Public interface for PersonDirectory identity enrichment.

    Called by PipelineOrchestrator after a successful face recognition to
    resolve ``person_id`` → ``person_name``.  Lookup is always fail-safe:
    unknown IDs return canonical UNKNOWN output, never an exception.
    """

    def get_person(self, person_id: str) -> PersonDirectoryOutput: ...


# ---------------------------------------------------------------------------
# Re-exports for convenient single-import access
# ---------------------------------------------------------------------------

__all__ = [
    "FaceDetectionInterface",
    "FaceRecognitionInterface",
    "FrameTransformationLayerInterface",
    "MotionDetectionInterface",
    "ObjectDetectionInterface",
    "PersonDirectoryInterface",
]
