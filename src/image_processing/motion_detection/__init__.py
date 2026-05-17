from __future__ import annotations

from image_processing.shared.contracts import BoundingBox

from .module import (
    FrameDifferencingMotionDetector,
    InputValidator,
    MotionDetectionAlgorithm,
    MotionDetectionConfig,
    MotionDetectionInput,
    MotionDetectionInterface,
    MotionDetectionManager,
    MotionDetectionResultInternal,
    MotionDecisionPolicy,
    MotionInputFrame,
    MotionMeasurementResult,
    MotionOutputBuilder,
    MotionResult,
    StubMotionDetectionAlgorithm,
)

__all__ = [
    # Public input / output models
    "BoundingBox",          # re-exported from image_processing.shared
    "MotionInputFrame",
    "MotionDetectionInput",
    "MotionResult",
    # Configuration
    "MotionDetectionConfig",
    # Public stage entry point and interface
    "MotionDetectionManager",
    "MotionDetectionInterface",
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
