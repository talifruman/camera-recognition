from __future__ import annotations

from .module import (
    BoundingBox,
    FramePacket,
    FrameDifferencingMotionDetector,
    InputValidator,
    MotionDetectionAlgorithm,
    MotionDetectionConfig,
    MotionDetectionInput,
    MotionDetectionManager,
    MotionDetectionResultInternal,
    MotionDecisionPolicy,
    MotionMeasurementResult,
    MotionOutputBuilder,
    MotionResult,
    StubMotionDetectionAlgorithm,
)

__all__ = [
    # Public input / output models
    "BoundingBox",
    "FramePacket",
    "MotionDetectionInput",
    "MotionResult",
    # Configuration
    "MotionDetectionConfig",
    # Public entry point
    "MotionDetectionManager",
    # Internal components (exposed for testing and advanced wiring)
    "InputValidator",
    "MotionDetectionAlgorithm",
    "FrameDifferencingMotionDetector",
    "StubMotionDetectionAlgorithm",
    "MotionDecisionPolicy",
    "MotionOutputBuilder",
    # Internal data structures (exposed for testing)
    "MotionMeasurementResult",
    "MotionDetectionResultInternal",
]
