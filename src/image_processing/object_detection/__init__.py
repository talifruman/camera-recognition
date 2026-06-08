from __future__ import annotations

from image_processing.shared.contracts import (
    GeometrySpec,
    Image,
    OutputImageType,
    PipelineStageInputContract,
    ResizePolicy,
)

from .module import (
    BoundingBox,
    FrameMetadata,
    ObjectDetectionInput,
    ObjectDetectionModule,
    PersonDetectionConfig,
    PersonDetectionResult,
    RawDetection,
    process,
)

__all__ = [
    # Primary public API
    "ObjectDetectionModule",
    "ObjectDetectionInput",
    "PersonDetectionResult",
    "BoundingBox",
    "PersonDetectionConfig",
    # Shared contract types (re-exported for convenience)
    "PipelineStageInputContract",
    "OutputImageType",
    "GeometrySpec",
    "ResizePolicy",
    "Image",
    # Internal / legacy — kept for backward compatibility
    "FrameMetadata",
    "RawDetection",
    "process",
]