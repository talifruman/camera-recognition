"""Recognition Pipeline Manager — public API package.

Public exports:
    RecognitionPipelineManager  — main entry point
    RecognitionPipelineOutput   — return type of process_frame()
    PersonResult                — one detected person with recognized faces
    RecognizedFaceResult        — one recognized face in full-frame coords

Interface exports (for dependency injection and testing):
    FrameTransformationLayerInterface
    ObjectDetectionInterface
    FaceDetectionInterface
    PersonDirectoryInterface
    MotionDetectionInterface     (re-exported from motion_detection)
    FaceRecognitionInterface     (re-exported from face_recognition)

NOT exported:
    PipelineResult              — internal accumulator
    PipelineOrchestrator        — internal component
    SpatialCoordinator          — internal component
    RecognitionPipelineInputValidator  — internal component
    RecognitionPipelineOutputBuilder   — internal component
"""

from __future__ import annotations

from .interfaces import (
    FaceDetectionInterface,
    FaceRecognitionInterface,
    FrameTransformationLayerInterface,
    MotionDetectionInterface,
    ObjectDetectionInterface,
    PersonDirectoryInterface,
)
from .recognition_pipeline_manager import RecognitionPipelineManager
from .types import PersonResult, RecognitionPipelineOutput, RecognizedFaceResult

__all__ = [
    # Public API
    "RecognitionPipelineManager",
    "RecognitionPipelineOutput",
    "PersonResult",
    "RecognizedFaceResult",
    # Interfaces (for injection and test doubles)
    "FaceDetectionInterface",
    "FaceRecognitionInterface",
    "FrameTransformationLayerInterface",
    "MotionDetectionInterface",
    "ObjectDetectionInterface",
    "PersonDirectoryInterface",
]
